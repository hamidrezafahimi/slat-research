"""
base_mono_vo.py
--------------------------------------------------------------------
Minimal but fully-functional **monocular visual-odometry** class.

Features
--------
* First two frames   : 8-point + cheirality baseline initialisation
* Later   frames     : EPnP pose (R|t) with known 3-D landmarks
* New landmarks      : DLT triangulation whenever a point is seen
                       in two consecutive frames
* Local BA           : Gauss–Newton (SciPy) over a sliding window
* Public API         :   BaseMonoVO(K, ba_window)
                         pts_cam, R_wc, t_wc = vo.do_vo(frame_uv)
                       where   frame_uv == [[id,u,v], …]

All coordinates remain up to an unknown **global scale** chosen to make
the very first baseline length = 1 internal unit.
--------------------------------------------------------------------
Author : ChatGPT demo • May-2025
"""

from __future__ import annotations
import numpy as np
import cv2
from scipy.optimize import least_squares


# ----------------- small helpers -----------------------------------

def rod_to_rmat(rvec: np.ndarray) -> np.ndarray:
    """Rodrigues (3,)  →  R (3,3)."""
    return cv2.Rodrigues(rvec.reshape(3, 1))[0]


def rmat_to_rod(R: np.ndarray) -> np.ndarray:
    """R (3,3)  →  Rodrigues (3,)."""
    return cv2.Rodrigues(R)[0].ravel()


# ===================================================================
#                           BaseMonoVO
# ===================================================================

class BaseMonoVO:
    """
    Pure monocular incremental VO that **returns landmark coordinates
    in the *current-camera* frame** at every call.

    Public method
    -------------
    pts_cam, R_wc, t_wc = vo.do_vo(frame_uv)

        frame_uv : (N,3) ndarray  [[id,u,v], …]
        pts_cam  : (M,4) ndarray  [[id,x,y,z], …]  (current camera)
        R_wc     : (3,3)          world → camera rotation
        t_wc     : (3,)           world → camera translation
    """

    # ------------------------------------------------------------------
    def __init__(self,
                 K: np.ndarray,
                 ba_window: int = 20):
        """
        Parameters
        ----------
        K          : (3,3) camera intrinsic matrix
        ba_window  : number of most-recent frames kept in BA
        """
        self.K = K.astype(np.float64)
        self.Nba = ba_window

        # trajectory & map containers
        self.poses: list[np.ndarray] = []            # 4×4, world→cam
        self.landmarks: dict[int, np.ndarray] = {}   # id → (3,)
        self.obs: dict[int, list[tuple[int, np.ndarray]]] = {}
        self.initialised = False

    # ================================================================
    #                       PUBLIC ENTRY
    # ================================================================
    def do_vo(self,
              frame_uv: np.ndarray):
        """
        One call per frame.

        frame_uv : (N,3)  [[id,u,v], …]   (float or int OK)

        Returns
        -------
        pts_cam : (M,4)  [[id,x,y,z], …]  in current camera frame
        R_wc, t_wc : latest pose
        """
        self._core_update(frame_uv)
        pts_cam = self._points_in_current_cam()
        R_wc, t_wc = self.current_pose()
        return pts_cam, R_wc, t_wc

    # ================================================================
    #                       INTERNAL PIPELINE
    # ================================================================
    def _core_update(self, frame_uv: np.ndarray):
        ids = frame_uv[:, 0].astype(int)
        uv  = frame_uv[:, 1:3].astype(np.float64)
        f_idx = len(self.poses)

        if not self.initialised:
            self._bootstrap_with_first_two_frames(ids, uv)
        else:
            self._track_and_pose(ids, uv, f_idx)

        # store raw 2-D observations
        for pid, xy in zip(ids, uv):
            self.obs.setdefault(pid, []).append((f_idx, xy))

        # optimise once we have ≥3 frames
        if self.initialised and len(self.poses) >= 3:
            self._bundle_adjust()

    # ----------------------------------------------------------------
    # 0.  Bootstrap (frames 0 & 1)  – eight-point + cheirality
    # ----------------------------------------------------------------
    def _bootstrap_with_first_two_frames(self,
                                         ids: np.ndarray,
                                         uv: np.ndarray):
        if len(self.poses) == 0:
            # store pose-0  (world frame)
            self.poses.append(np.eye(4))
            self.frame0_ids, self.frame0_uv = ids, uv
            return

        # frame-1 arrives
        ids0, uv0 = self.frame0_ids, self.frame0_uv
        mask = np.isin(ids0, ids)
        ids0c, uv0c = ids0[mask], uv0[mask]
        idx1 = np.nonzero(np.isin(ids, ids0c))[0]
        uv1c = uv[idx1]

        # essential matrix
        E, inl = cv2.findEssentialMat(
            uv0c, uv1c, self.K,
            method=cv2.RANSAC,
            prob=0.999, threshold=1.0
        )
        inl = inl.ravel().astype(bool)
        _, R, t_hat, _ = cv2.recoverPose(E,
                                         uv0c[inl], uv1c[inl], self.K)

        # choose baseline |t| = 1 (internal units)
        t = t_hat.ravel() / np.linalg.norm(t_hat)

        T1 = np.eye(4)
        T1[:3, :3] = R
        T1[:3, 3] = t
        self.poses.append(T1)
        self.initialised = True

        # seed landmarks
        common_ids = ids0c[inl]
        self._triangulate_new(common_ids, uv0c[inl],
                              common_ids, uv1c[inl],
                              T0=self.poses[0], T1=T1)

    # ----------------------------------------------------------------
    # 1.  Normal frame (PnP pose + triangulate new points)
    # ----------------------------------------------------------------
    def _track_and_pose(self,
                        ids: np.ndarray,
                        uv: np.ndarray,
                        f_idx: int):
        # collect 3-D-2-D correspondences
        obj, img = [], []
        for pid, xy in zip(ids, uv):
            if pid in self.landmarks:
                obj.append(self.landmarks[pid])
                img.append(xy)

        if len(obj) < 4:
            raise RuntimeError("Not enough 3-D points for PnP")

        obj = np.asarray(obj, np.float32)
        img = np.asarray(img, np.float32)

        # EPnP pose
        ok, rvec, tvec = cv2.solvePnP(
            obj, img, self.K, None,
            flags=cv2.SOLVEPNP_EPNP
        )
        if not ok:
            raise RuntimeError("solvePnP failed")

        R = rod_to_rmat(rvec)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = tvec.ravel()
        self.poses.append(T)

        # triangulate brand-new points seen in f-1 and f
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
                np.array(prev_ids), np.array([uv[np.where(ids == pid)[0][0]]
                                              for pid in prev_ids]),
                T0=self.poses[prev_idx], T1=T
            )

    # ----------------------------------------------------------------
    # triangulation helper (DLT linear)
    # ----------------------------------------------------------------
    def _triangulate_new(self,
                         ids0: np.ndarray, uv0: np.ndarray,
                         ids1: np.ndarray, uv1: np.ndarray,
                         T0: np.ndarray, T1: np.ndarray):
        if len(ids0) == 0:
            return
        P0 = self.K @ T0[:3, :]
        P1 = self.K @ T1[:3, :]
        pts4 = cv2.triangulatePoints(
            P0, P1, uv0.astype(np.float32).T, uv1.astype(np.float32).T)
        pts3 = (pts4[:3] / pts4[3]).T
        finite = np.isfinite(pts3).all(axis=1)
        for pid, X, ok in zip(ids0, pts3, finite):
            if ok:
                self.landmarks.setdefault(int(pid), X)
        for pid, X in zip(ids0, pts3):
            self.landmarks.setdefault(int(pid), X)

    # ----------------------------------------------------------------
    # 2.  Local bundle adjustment (Gauss-Newton, sliding window)
    # ----------------------------------------------------------------
    def _bundle_adjust(self):
        last = len(self.poses) - 1
        first = max(0, last - self.Nba + 1)
        frame_ids = list(range(first, last + 1))
        f2idx = {f: i for i, f in enumerate(frame_ids)}

        # pack pose parameters
        pose_vec = []
        for f in frame_ids:
            T = self.poses[f]
            pose_vec.extend(rmat_to_rod(T[:3, :3]))
            pose_vec.extend(T[:3, 3])
        pose_vec = np.array(pose_vec)

        # landmarks in window
        lm_ids = [pid for pid, obs in self.obs.items()
                  if any(first <= f <= last for f, _ in obs)]
        lm_vec = np.concatenate([self.landmarks[pid] for pid in lm_ids])

        # observation tuples
        obs_tuples = []
        for j, pid in enumerate(lm_ids):
            for f, uv in self.obs[pid]:
                if first <= f <= last:
                    obs_tuples.append((f2idx[f], j, uv))

        def unpack(x):
            p, l = x[:pose_vec.size], x[pose_vec.size:]
            return p, l.reshape((-1, 3))

        def residuals(x):
            pvec, lms = unpack(x)
            res = []
            Rm, tm = [], []
            for i in range(len(frame_ids)):
                r = pvec[6*i:6*i+3]
                t = pvec[6*i+3:6*i+6]
                Rm.append(rod_to_rmat(r))
                tm.append(t)
            for fidx, j, uv in obs_tuples:
                X = lms[j]
                x_cam = Rm[fidx] @ X + tm[fidx]
                u_proj = self.K[0, 0] * x_cam[0] / x_cam[2] + self.K[0, 2]
                v_proj = self.K[1, 1] * x_cam[1] / x_cam[2] + self.K[1, 2]
                res.extend([u_proj - uv[0], v_proj - uv[1]])
            return np.asarray(res)

        x0 = np.concatenate([pose_vec, lm_vec])
        if x0.size == 0:
            return
        res = least_squares(residuals, x0, verbose=0, max_nfev=10)
        pose_opt, lm_opt = unpack(res.x)

        # write back
        for i, f in enumerate(frame_ids):
            r = pose_opt[6*i:6*i+3]
            t = pose_opt[6*i+3:6*i+6]
            R = rod_to_rmat(r)
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3]  = t
            self.poses[f] = T
        for pid, xyz in zip(lm_ids, lm_opt):
            self.landmarks[pid] = xyz

    # ----------------------------------------------------------------
    # Helpers to get output in camera frame
    # ----------------------------------------------------------------
    def current_pose(self):
        if not self.poses:
            raise RuntimeError("VO not initialised")
        T = self.poses[-1]
        return T[:3, :3].copy(), T[:3, 3].copy()

    def _points_in_current_cam(self):
        R_wc, t_wc = self.current_pose()
        rows = []
        for pid, Xw in self.landmarks.items():
            Xc = R_wc @ Xw + t_wc
            rows.append([pid, *Xc])
        return np.asarray(rows, np.float64)
