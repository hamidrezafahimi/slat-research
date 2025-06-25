"""
MonoStreamVO – monocular visual‑odometry with optional range priors
==================================================================

**31‑May‑2025 – added absolute‑distance support**

You can now pass *per‑iteration* Euclidean distances (in **metres**) to any
subset of current keypoints – for example from a laser‑rangefinder, depth
sensor, or known calibration target.  Format:

```python
# same order as the frame’s 2‑D keypoints; use None when not available
range_info = np.array([[id0, None], [id2, 7.3], …], dtype=object)
vo.do_vo(frame_uv, range_info)  # new optional 2nd arg
```

The distances are injected as **additional residuals** in the sliding‑window
bundle‑adjustment (BA):

```
res_depth = (‖X_cam‖ – d_measured) / depth_sigma
```

so they act as soft constraints that:

* resolve the global *scale* ambiguity of monocular VO,
* suppress scale drift over time, and
* provide another outlier test (depth gate + robust Huber loss).

---
"""

from __future__ import annotations
import numpy as np
import cv2
from scipy.optimize import least_squares
from typing import Dict, List, Tuple, Optional, Union

# ─────────── Rodrigues helpers ───────────────────────────────────────────

def rodrigues_to_rmat(rvec: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(rvec.reshape(3, 1)); return R

def rmat_to_rodrigues(R: np.ndarray) -> np.ndarray:
    r, _ = cv2.Rodrigues(R); return r.ravel()

# ════════════════════════════════════════════════════════════════════════
class MonoStreamVO:
    """Incremental monocular VO with sliding‑window BA, outlier guards,
    **and optional range (distance) observations**.
    """

    # Tunables -----------------------------------------------------------
    reproj_thresh: float = 2.0   # [px] gate when triangulating
    depth_factor:  float = 10.0  # ×median gate when triangulating/pruning
    depth_sigma:   float = 0.10  # [m] 1‑σ noise of external range sensor
    ba_loss: str = "huber"
    ba_fscale: float = 3.0       # Huber threshold [px] / [m]

    # -------------------------------------------------------------------
    def __init__(self, K: np.ndarray, ba_win: int = 5):
        self.K = K.astype(np.float64)
        self.ba_win = ba_win

        self.poses: List[np.ndarray] = []            # 4×4 world→cam
        self.landmarks: Dict[int, np.ndarray] = {}   # id → (3,)
        self.obs: Dict[int, List[Tuple[int, np.ndarray]]] = {}
        self.range_obs: Dict[int, List[Tuple[int, float]]] = {}

        self.initialised = False

    # ───────────── helper ───────────────────────────────────────────────
    def current_pose(self):
        if not self.poses:
            raise RuntimeError("VO has no pose yet.")
        T = self.poses[-1]; return T[:3, :3].copy(), T[:3, 3].copy()

    # ═════════════════ main entry ═══════════════════════════════════════
    def do_vo(self, frame_uv: np.ndarray,
              range_info: Optional[np.ndarray] = None):
        """Process one frame.

        Parameters
        ----------
        frame_uv : (N,3) ``[[id,u,v],…]`` pixel measurements
        range_info : (N,2) ``[[id,d],…]``  *object* dtype, distance in metres
                      (use *None* when no range for that id).
        """
        ids = frame_uv[:, 0].astype(int)
        uv  = frame_uv[:, 1:3].astype(np.float64)
        f_idx = len(self.poses)

        if not self.initialised:
            self._bootstrap_with_first_two_frames(ids, uv)
        else:
            self._track_and_pose(ids, uv, f_idx)

        # log 2‑D observations every frame
        for pid, xy in zip(ids, uv):
            self.obs.setdefault(pid, []).append((f_idx, xy))

        # log range observations if provided
        if range_info is not None:
            for pid, dist in range_info:
                if dist is None:  # skip missing entries
                    continue
                pid = int(pid)
                self.range_obs.setdefault(pid, []).append((f_idx, float(dist)))

        # BA + pruning ---------------------------------------------------
        if self.initialised and len(self.poses) >= 3:
            self._bundle_adjust(); self._prune_far_landmarks()

        # output landmarks in *current‑camera* frame ---------------------
        R_wc, t_wc = self.current_pose()
        return np.asarray([[pid, *(R_wc @ X + t_wc)]
                           for pid, X in self.landmarks.items()], dtype=np.float64)

    # ───────────────────── bootstrap ────────────────────────────────────
    def _bootstrap_with_first_two_frames(self, ids1, uv1):
        if not self.poses:                        # first frame
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
        t = t_hat.ravel() / np.linalg.norm(t_hat)  # arbitrary initial scale

        T1 = np.eye(4); T1[:3, :3], T1[:3, 3] = R, t
        self.poses.append(T1)

        self._triangulate_new(ids0c[inl], uv0c[inl],
                              ids1c[inl], uv1c[inl],
                              self.poses[0], T1)
        self.initialised = True

    # ───────────────────── pose & tracking ─────────────────────────────
    def _track_and_pose(self, ids, uv, f_idx):
        obj, img = [], []
        for pid, xy in zip(ids, uv):
            if pid in self.landmarks:
                obj.append(self.landmarks[pid]); img.append(xy)
        obj, img = np.asarray(obj, np.float32), np.asarray(img, np.float32)

        if len(obj) < 4:
            self.poses.append(self.poses[-1].copy()); return

        ok, rvec, tvec = cv2.solvePnP(obj, img, self.K, None, flags=cv2.SOLVEPNP_EPNP)
        if not ok:
            self.poses.append(self.poses[-1].copy()); return

        R, t = rodrigues_to_rmat(rvec), tvec.ravel()
        T = np.eye(4); T[:3, :3], T[:3, 3] = R, t; self.poses.append(T)

        # opportunistic triangulation for brand‑new points
        prev_idx = f_idx - 1
        new_pairs = [(pid, puv, cuv) for pid, cuv in zip(ids, uv)
                     if pid not in self.landmarks
                     for pf, puv in self.obs.get(pid, []) if pf == prev_idx]
        if new_pairs:
            pids, uv0, uv1 = zip(*new_pairs)
            self._triangulate_new(np.array(pids), np.array(uv0),
                                  np.array(pids), np.array(uv1),
                                  self.poses[prev_idx], T)

    # ───────────────────── triangulation ───────────────────────────────
    def _triangulate_new(self, ids0, uv0, ids1, uv1, T0, T1):
        P0, P1 = self.K @ T0[:3], self.K @ T1[:3]
        pts4 = cv2.triangulatePoints(P0, P1, uv0.T, uv1.T)
        pts3 = (pts4[:3] / pts4[3]).T

        if self.landmarks:
            med_dist = np.median(np.linalg.norm(list(self.landmarks.values()), axis=1))
            cutoff = self.depth_factor * med_dist
        else:
            cutoff = np.inf

        for pid, X, u0, u1 in zip(ids0, pts3, uv0, uv1):
            x0 = P0 @ np.append(X, 1.0); x0 = x0[:2]/x0[2]
            x1 = P1 @ np.append(X, 1.0); x1 = x1[:2]/x1[2]
            if (np.linalg.norm(x0-u0) > self.reproj_thresh or
                np.linalg.norm(x1-u1) > self.reproj_thresh or
                np.linalg.norm(X) > cutoff):
                continue
            self.landmarks[int(pid)] = X

    # ───────────────────── pruning ─────────────────────────────────────
    def _prune_far_landmarks(self):
        if not self.landmarks: return
        dists = np.array([np.linalg.norm(X) for X in self.landmarks.values()])
        med, cutoff = np.median(dists), self.depth_factor * np.median(dists)
        for pid, X in list(self.landmarks.items()):
            if np.linalg.norm(X) > cutoff:
                del self.landmarks[pid]; self.obs.pop(pid, None); self.range_obs.pop(pid, None)

    # ───────────────────── bundle‑adjustment ───────────────────────────
    def _bundle_adjust(self):
        last, first = len(self.poses)-1, max(0, len(self.poses)-self.ba_win)
        frame_ids = list(range(first, last+1)); f2idx = {f:i for i,f in enumerate(frame_ids)}

        # pose params
        pose_params = np.concatenate([np.hstack([rmat_to_rodrigues(self.poses[f][:3,:3]),
                                                 self.poses[f][:3,3]])
                                       for f in frame_ids])
        # landmarks inside window
        lm_ids = [pid for pid,obs in self.obs.items()
                  if pid in self.landmarks and any(first<=f<=last for f,_ in obs)]
        if not lm_ids: return
        lm_params = np.concatenate([self.landmarks[pid] for pid in lm_ids])

        # 2‑D and range observations
        img_obs = [(f2idx[f], j, uv) for j,pid in enumerate(lm_ids)
                    for f,uv in self.obs[pid] if first<=f<=last]
        rng_obs = [(f2idx[f], j, d) for j,pid in enumerate(lm_ids)
                    for f,d in self.range_obs.get(pid, []) if first<=f<=last]

        def unpack(x):
            p = x[:pose_params.size]
            l = x[pose_params.size:].reshape(-1,3)
            return p,l

        def residuals(x):
            p,l = unpack(x)
            Rs,ts=[],[]
            for i in range(len(frame_ids)):
                r,t = p[6*i:6*i+3], p[6*i+3:6*i+6]
                Rs.append(rodrigues_to_rmat(r)); ts.append(t)
            res=[]
            # reprojection residuals (pixels)
            for fidx,j,uv in img_obs:
                X = l[j]; x_cam = Rs[fidx]@X + ts[fidx]; xp = self.K @ x_cam
                res.extend([xp[0]/xp[2]-uv[0], xp[1]/xp[2]-uv[1]])
            # range residuals (metres)
            for fidx,j,d_meas in rng_obs:
                X = l[j]; dist = np.linalg.norm(Rs[fidx]@X + ts[fidx])
                res.append((dist - d_meas) / self.depth_sigma)  # normalise by σ
            return np.array(res)

        x0 = np.concatenate([pose_params,lm_params])
        res = least_squares(residuals, x0, loss=self.ba_loss, f_scale=self.ba_fscale,
                            verbose=0, max_nfev=25)
        p_opt,l_opt = unpack(res.x)

        # write‑back poses
        for i,f in enumerate(frame_ids):
            r,t = p_opt[6*i:6*i+3], p_opt[6*i+3:6*i+6]
            R = rodrigues_to_rmat(r); T=np.eye(4); T[:3,:3],T[:3,3] = R,t; self.poses[f]=T
        # write‑back landmarks
        for pid,X in zip(lm_ids,l_opt): self.landmarks[pid]=X
