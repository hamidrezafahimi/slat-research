"""
MonoStreamVO – monocular visual-odometry pipeline
=================================================

Updated 31‑May‑2025
-------------------
* **Dynamic keypoint set**: points can appear/disappear freely.
* **Outlier suppression**:
  1. Reprojection gate (≤ ``reproj_thresh`` px) during triangulation.
  2. Depth gate (≤ ``depth_factor×median``) — *disabled* when there are **no** landmarks yet so that boot‑strapping works!
  3. Robust BA (`loss='huber'`).

Call ``do_vo(frame_uv)`` once per image; it returns landmarks in the *current‑camera* frame: ``[[id, x, y, z], …]``.
"""

from __future__ import annotations
import numpy as np
import cv2
from scipy.optimize import least_squares

# ─────────────── Rodrigues helpers ───────────────────────────────────────

def rodrigues_to_rmat(rvec: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(rvec.reshape(3, 1)); return R

def rmat_to_rodrigues(R: np.ndarray) -> np.ndarray:
    r, _ = cv2.Rodrigues(R); return r.ravel()

# ════════════════════════════════════════════════════════════════════════
class MonoStreamVO:
    """Incremental monocular VO with local BA and real‑time outlier guards."""

    # tunables -----------------------------------------------------------
    reproj_thresh: float = 2.0   # [px]
    depth_factor:  float = 10.0  # ×median
    ba_loss: str = "huber"
    ba_fscale: float = 3.0       # Huber switch [px]

    # -------------------------------------------------------------------
    def __init__(self, K: np.ndarray, ba_win: int = 5):
        self.K = K.astype(np.float64)
        self.ba_win = ba_win
        self.poses: list[np.ndarray] = []           # 4×4 world→cam
        self.landmarks: dict[int, np.ndarray] = {}  # id → (3,)
        self.obs: dict[int, list[tuple[int, np.ndarray]]] = {}
        self.initialised = False

    # ───────────────── helper ───────────────────────────────────────────
    def current_pose(self):
        if not self.poses:
            raise RuntimeError("VO has no pose yet.")
        T = self.poses[-1]; return T[:3, :3].copy(), T[:3, 3].copy()

    # ═════════════════ VO main entry ════════════════════════════════════
    def do_vo(self, frame_uv: np.ndarray):
        ids = frame_uv[:, 0].astype(int)
        uv  = frame_uv[:, 1:3].astype(np.float64)
        f_idx = len(self.poses)

        if not self.initialised:
            self._bootstrap_with_first_two_frames(ids, uv)
        else:
            self._track_and_pose(ids, uv, f_idx)

        # log observations
        for pid, xy in zip(ids, uv):
            self.obs.setdefault(pid, []).append((f_idx, xy))

        if self.initialised and len(self.poses) >= 3:
            self._bundle_adjust(); self._prune_far_landmarks()

        R_wc, t_wc = self.current_pose()
        return np.asarray([[pid, *(R_wc @ X + t_wc)]
                           for pid, X in self.landmarks.items()], dtype=np.float64)

    # ───────────────── bootstrap ────────────────────────────────────────
    def _bootstrap_with_first_two_frames(self, ids1, uv1):
        if not self.poses:                      # very first frame
            self.poses.append(np.eye(4))
            self.frame0_ids, self.frame0_uv = ids1, uv1; return

        ids0, uv0 = self.frame0_ids, self.frame0_uv
        mask = np.isin(ids0, ids1)
        ids0c, uv0c = ids0[mask], uv0[mask]
        idx1 = np.nonzero(np.isin(ids1, ids0c))[0]
        ids1c, uv1c = ids1[idx1], uv1[idx1]
        if len(ids0c) < 8: raise RuntimeError("Need ≥8 matches for init.")

        E, inl = cv2.findEssentialMat(uv0c, uv1c, self.K, cv2.RANSAC, 0.999, 1.0)
        inl = inl.ravel().astype(bool)
        _, R, t_hat, _ = cv2.recoverPose(E, uv0c[inl], uv1c[inl], self.K)
        t = t_hat.ravel() / np.linalg.norm(t_hat)

        T1 = np.eye(4); T1[:3, :3], T1[:3, 3] = R, t
        self.poses.append(T1)

        self._triangulate_new(ids0c[inl], uv0c[inl],
                              ids1c[inl], uv1c[inl],
                              self.poses[0], T1)
        self.initialised = True

    # ───────────────── pose tracking ────────────────────────────────────
    def _track_and_pose(self, ids, uv, f_idx):
        obj, img = [], []
        for pid, xy in zip(ids, uv):
            if pid in self.landmarks: obj.append(self.landmarks[pid]); img.append(xy)
        obj, img = np.asarray(obj, np.float32), np.asarray(img, np.float32)

        if len(obj) < 4: self.poses.append(self.poses[-1].copy()); return

        ok, rvec, tvec = cv2.solvePnP(obj, img, self.K, None, flags=cv2.SOLVEPNP_EPNP)
        if not ok: self.poses.append(self.poses[-1].copy()); return

        R, t = rodrigues_to_rmat(rvec), tvec.ravel()
        T = np.eye(4); T[:3, :3], T[:3, 3] = R, t; self.poses.append(T)

        prev_idx = f_idx - 1
        new_pairs = [(pid, puv, cuv) for pid, cuv in zip(ids, uv)
                     if pid not in self.landmarks
                     for pf, puv in self.obs.get(pid, []) if pf == prev_idx]
        if new_pairs:
            pids, uv0, uv1 = zip(*new_pairs)
            self._triangulate_new(np.array(pids), np.array(uv0),
                                  np.array(pids), np.array(uv1),
                                  self.poses[prev_idx], T)

    # ───────────────── triangulation ────────────────────────────────────
    def _triangulate_new(self, ids0, uv0, ids1, uv1, T0, T1):
        P0, P1 = self.K @ T0[:3], self.K @ T1[:3]
        pts4 = cv2.triangulatePoints(P0, P1, uv0.T, uv1.T)
        pts3 = (pts4[:3] / pts4[3]).T

        if self.landmarks:
            dists_existing = np.linalg.norm(list(self.landmarks.values()), axis=1)
            med_dist = np.median(dists_existing)
            depth_cutoff = self.depth_factor * med_dist
        else:
            depth_cutoff = np.inf  # first batch – accept all depths

        for pid, X, u0, u1 in zip(ids0, pts3, uv0, uv1):
            x0 = P0 @ np.append(X, 1.0); x0 = x0[:2]/x0[2]
            x1 = P1 @ np.append(X, 1.0); x1 = x1[:2]/x1[2]
            if (np.linalg.norm(x0-u0) > self.reproj_thresh or
                np.linalg.norm(x1-u1) > self.reproj_thresh):
                continue
            if np.linalg.norm(X) > depth_cutoff: continue
            self.landmarks[int(pid)] = X

    # ───────────────── pruning ─────────────────────────────────────────
    def _prune_far_landmarks(self):
        if not self.landmarks: return
        dists = np.array([np.linalg.norm(X) for X in self.landmarks.values()])
        med = np.median(dists); cutoff = self.depth_factor * med
        for pid, X in list(self.landmarks.items()):
            if np.linalg.norm(X) > cutoff:
                del self.landmarks[pid]; self.obs.pop(pid, None)

    # ───────────────── bundle‑adjustment ───────────────────────────────
    def _bundle_adjust(self):
        last, first = len(self.poses)-1, max(0, len(self.poses)-self.ba_win)
        frame_ids = list(range(first, last+1)); f2idx = {f:i for i,f in enumerate(frame_ids)}

        pose_params = np.concatenate([np.hstack([rmat_to_rodrigues(T[:3,:3]), T[:3,3]])
                                      for T in (self.poses[f] for f in frame_ids)])
        lm_ids = [pid for pid,obs in self.obs.items()
                  if pid in self.landmarks and any(first<=f<=last for f,_ in obs)]
        if not lm_ids: return
        lm_params = np.concatenate([self.landmarks[pid] for pid in lm_ids])
        obs_tuples = [(f2idx[f], j, uv) for j,pid in enumerate(lm_ids)
                       for f,uv in self.obs[pid] if first<=f<=last]

        def unpack(x):
            p = x[:pose_params.size]; l = x[pose_params.size: ].reshape(-1,3); return p,l
        def residuals(x):
            p,l = unpack(x)
            Rs,ts=[],[]
            for i in range(len(frame_ids)):
                r,t = p[6*i:6*i+3], p[6*i+3:6*i+6]
                Rs.append(rodrigues_to_rmat(r)); ts.append(t)
            res=[]
            for fidx,j,uv in obs_tuples:
                X = l[j]; x_cam = Rs[fidx]@X + ts[fidx]
                xp = self.K @ x_cam; res.extend([xp[0]/xp[2]-uv[0], xp[1]/xp[2]-uv[1]])
            return np.array(res)

        x0 = np.concatenate([pose_params,lm_params])
        res = least_squares(residuals, x0, loss=self.ba_loss, f_scale=self.ba_fscale, max_nfev=15)
        p_opt,l_opt = unpack(res.x)
        for i,f in enumerate(frame_ids):
            r,t = p_opt[6*i:6*i+3], p_opt[6*i+3:6*i+6]
            R = rodrigues_to_rmat(r); T=np.eye(4); T[:3,:3],T[:3,3] = R,t; self.poses[f]=T
        for pid,X in zip(lm_ids,l_opt): self.landmarks[pid]=X
