"""
Minimal monocular visual-odometry pipeline with
  • Essential-matrix baseline initialisation
  • PnP pose for every later frame
  • new-point triangulation
  • sliding-window bundle-adjustment (SciPy least-squares)

Author: ChatGPT demo • May-2025
"""

from __future__ import annotations
import numpy as np
import cv2
from scipy.optimize import least_squares


# -------------------------------------------------------------
def rodrigues_to_rmat(rvec: np.ndarray) -> np.ndarray:
    """(3,) -> (3,3) rotation matrix."""
    R, _ = cv2.Rodrigues(rvec.reshape(3, 1))
    return R


def rmat_to_rodrigues(R: np.ndarray) -> np.ndarray:
    """(3,3) -> (3,) rotation vector."""
    r, _ = cv2.Rodrigues(R)
    return r.ravel()


# -------------------------------------------------------------
class MonoStreamVO:
    """
    Monocular (single-camera) incremental VO with local bundle-adjustment.

    Call `do_vo(arr)` once per frame, where
        arr = [[id, u, v], …]   float32
    returns
        [[id, x, y, z], …]      float64 – current landmark positions
    """

    def __init__(self, K: np.ndarray, ba_win: int = 5):
        self.K = K.astype(np.float64)
        self.poses: list[np.ndarray] = []          # list of 4×4 world→cam
        self.landmarks: dict[int, np.ndarray] = {} # id → (3,)
        self.obs: dict[int, list[tuple[int, np.ndarray]]] = {}  # id → [(frame, uv)]
        self.ba_win = ba_win                       # #frames in BA window
        self.initialised = False                   # True after first TWO frames

    # ---------------------------------------------------------
   # tiny helper – world → camera
    @staticmethod
    def _w2c(R_wc: np.ndarray, t_wc: np.ndarray, Xw: np.ndarray) -> np.ndarray:
        return (R_wc @ Xw.T + t_wc[:, None]).T      # (N,3)

    # ------------------------------------------------------------------
    def current_pose(self):
        """
        Latest pose (world → current camera).
        Returns
        -------
        R_wc : (3,3)  rotation matrix
        t_wc : (3,)   translation vector
        """
        if not self.poses:
            raise RuntimeError("VO has no pose yet.")
        T = self.poses[-1]
        return T[:3, :3].copy(), T[:3, 3].copy()

    # ------------------------------------------------------------------
    def do_vo(self, frame_uv: np.ndarray):
        """
        One VO update.

        Parameters
        ----------
        frame_uv : (N,3)  [[id, u, v], …]  pixel coords

        Returns
        -------
        pts_cam : (M,4)  [[id, x, y, z], …]  **in *current-camera* frame**
        R_wc, t_wc : latest pose (world → camera)
        """
        # -- existing internal pipeline ---------------------------------
        ids = frame_uv[:, 0].astype(int)
        uv  = frame_uv[:, 1:3].astype(np.float64)
        f_idx = len(self.poses)

        if not self.initialised:
            self._bootstrap_with_first_two_frames(ids, uv)
        else:
            self._track_and_pose(ids, uv, f_idx)

        for pid, xy in zip(ids, uv):
            self.obs.setdefault(pid, []).append((f_idx, xy))

        if self.initialised and len(self.poses) >= 3:
            self._bundle_adjust()

        # -- *** new part: return points in camera frame *** -------------
        R_wc, t_wc = self.current_pose()

        rows = []
        for pid, Xw in self.landmarks.items():
            Xc = R_wc @ Xw + t_wc          # world → camera
            rows.append([pid, *Xc])

        pts_cam = np.asarray(rows, dtype=np.float64)
        # return pts_cam, R_wc, t_wc   # <-- keep pose handy for plotting
        return pts_cam

    # ================================================================ #
    #  internal helpers                                                #
    # ================================================================ #

    def _bootstrap_with_first_two_frames(self, ids1: np.ndarray,
                                         uv1: np.ndarray) -> None:
        """
        Called once on the first invocation, *again* on the second frame.
        After that `self.initialised` becomes True.
        """
        if len(self.poses) == 0:
            # store frame-0 pose (origin) and wait for frame-1
            T0 = np.eye(4)
            self.poses.append(T0)
            self.frame0_ids, self.frame0_uv = ids1, uv1
            return

        # ------------ now we have frame-0 and frame-1 -----------------
        ids0, uv0 = self.frame0_ids, self.frame0_uv

        # keep *common* ids only
        mask = np.isin(ids0, ids1)
        ids0c, uv0c = ids0[mask], uv0[mask]
        idx1 = np.nonzero(np.isin(ids1, ids0c))[0]
        ids1c, uv1c = ids1[idx1], uv1[idx1]

        E, inl = cv2.findEssentialMat(
            uv0c, uv1c, self.K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0
        )
        inl = inl.ravel().astype(bool)
        _, R, t_hat, _ = cv2.recoverPose(
            E, uv0c[inl], uv1c[inl], self.K
        )

        # choose baseline length = 1 (internal unit)
        t = t_hat.ravel() / np.linalg.norm(t_hat)

        # world frame = cam-0, cam-1 pose:
        T1 = np.eye(4)
        T1[:3, :3] = R
        T1[:3, 3]  = t
        self.poses.append(T1)

        # triangulate initial landmarks
        self._triangulate_new(ids0c[inl], uv0c[inl],
                              ids1c[inl], uv1c[inl],
                              T0=self.poses[0], T1=T1)

        self.initialised = True
        print(" [bootstrap] baseline initialised – internal scale fixed")

    # ---------------------------------------------------------------
    def _track_and_pose(self, ids: np.ndarray, uv: np.ndarray,
                        f_idx: int) -> None:
        """
        For frames #2, #3, … : PnP with already-known landmarks.
        """
        # collect 3-D ↔ 2-D correspondences
        obj, img = [], []
        for pid, xy in zip(ids, uv):
            if pid in self.landmarks:
                obj.append(self.landmarks[pid])
                img.append(xy)
        obj = np.asarray(obj, np.float32)
        img = np.asarray(img, np.float32)

        if len(obj) < 4:
            raise RuntimeError("PnP needs ≥4 known points – tracking failed?")

        ok, rvec, tvec = cv2.solvePnP(
            obj, img, self.K, None,
            flags=cv2.SOLVEPNP_EPNP
        )
        if not ok:
            raise RuntimeError("solvePnP failed")

        R = rodrigues_to_rmat(rvec)
        t = tvec.ravel()

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3]  = t
        self.poses.append(T)

        # triangulate brand-new points that also existed in previous frame
        prev_ids, prev_uv = [], []
        prev_frame_idx = f_idx - 1
        for pid, xy in zip(ids, uv):
            if pid not in self.landmarks:
                # was it in (f_idx-1) ? …
                for pidx, puv in self.obs.get(pid, []):
                    if pidx == prev_frame_idx:
                        prev_ids.append(pid)
                        prev_uv.append(puv)
        if prev_ids:
            self._triangulate_new(
                np.array(prev_ids), np.array(prev_uv),
                np.array(prev_ids), np.array([uv[np.where(ids==pid)[0][0]]
                                              for pid in prev_ids]),
                T0=self.poses[prev_frame_idx], T1=T
            )

    # ---------------------------------------------------------------
    def _triangulate_new(self,
                         ids0: np.ndarray, uv0: np.ndarray,
                         ids1: np.ndarray, uv1: np.ndarray,
                         T0: np.ndarray, T1: np.ndarray) -> None:
        """DLT triangulation for matching points seen in two frames."""
        P0 = self.K @ T0[:3, :]
        P1 = self.K @ T1[:3, :]
        pts4 = cv2.triangulatePoints(P0, P1, uv0.T, uv1.T)
        pts3 = (pts4[:3] / pts4[3]).T
        for pid, X in zip(ids0, pts3):
            # if still absent (might have been triangulated earlier)
            self.landmarks.setdefault(int(pid), X)

    # ---------------------------------------------------------------
    def _bundle_adjust(self) -> None:
        """
        Small Gauss–Newton BA over the last `ba_win` frames
        (poses) and any landmarks observed inside that window.
        """
        # ------------ gather window data --------------------------
        last = len(self.poses) - 1
        first = max(0, last - self.ba_win + 1)
        frame_ids = list(range(first, last + 1))
        f2idx = {f: i for i, f in enumerate(frame_ids)}

        # poses: param = [rvec, t] per frame
        pose_params = []
        for f in frame_ids:
            T = self.poses[f]
            pose_params.append(rmat_to_rodrigues(T[:3, :3]))
            pose_params.append(T[:3, 3])
        pose_params = np.concatenate(pose_params)          # (6·F,)

        # landmarks in window
        lm_ids = []
        for pid, obss in self.obs.items():
            if any(first <= f < last+1 for f, _ in obss):
                lm_ids.append(pid)
        lm_params = np.concatenate([self.landmarks[pid] for pid in lm_ids])

        # observations
        obs_tuples = []   # (f_local_idx, lm_local_idx, uv)
        for j, pid in enumerate(lm_ids):
            for f, uv in self.obs[pid]:
                if first <= f <= last:
                    obs_tuples.append((f2idx[f], j, uv))

        def pack_params(pose_vec, lm_vec):
            return np.concatenate([pose_vec, lm_vec])

        def unpack_params(x):
            pose_vec = x[:pose_params.size]
            lm_vec   = x[pose_params.size:]
            return pose_vec, lm_vec

        # ------------- residual function --------------------------
        def residuals(x):
            pose_vec, lm_vec = unpack_params(x)
            res = []
            # per-frame pose mats
            Rmats, tvecs = [], []
            for i in range(len(frame_ids)):
                r = pose_vec[6*i:6*i+3]
                t = pose_vec[6*i+3:6*i+6]
                Rmats.append(rodrigues_to_rmat(r))
                tvecs.append(t)
            # per-landmark coords
            lms = lm_vec.reshape((-1, 3))

            for fidx, j, uv in obs_tuples:
                R = Rmats[fidx]
                t = tvecs[fidx]
                X = lms[j]
                x_cam = R @ X + t
                x_proj = self.K @ x_cam
                u = x_proj[0] / x_proj[2]
                v = x_proj[1] / x_proj[2]
                res.extend([u - uv[0], v - uv[1]])
            return np.array(res)

        x0 = pack_params(pose_params, lm_params)
        if x0.size == 0:
            return  # nothing to optimise (unlikely)
        res = least_squares(residuals, x0, verbose=0, max_nfev=10)
        pose_opt, lm_opt = unpack_params(res.x)

        # ------------- write back results -------------------------
        # poses
        for i, f in enumerate(frame_ids):
            r = pose_opt[6*i:6*i+3]
            t = pose_opt[6*i+3:6*i+6]
            R = rodrigues_to_rmat(r)
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3]  = t
            self.poses[f] = T
        # landmarks
        for pid, xyz in zip(lm_ids, lm_opt.reshape((-1, 3))):
            self.landmarks[pid] = xyz

