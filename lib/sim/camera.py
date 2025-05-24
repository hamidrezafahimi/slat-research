import math
from typing import Tuple

import cv2
import numpy as np


class PinholeCamera:
    """
    Ideal pinhole‑camera model (no lens distortion) with optional on‑screen preview.

    Points and projections include a unique identifier per point.

    Parameters
    ----------
    fx, fy : float
        Focal lengths in **pixels**.
    cx, cy : float
        Principal‑point coordinates in **pixels**.
    image_shape : tuple (width, height), optional
        Size of the preview canvas.  If omitted a canvas of
        ``(int(2*cx), int(2*cy))`` is created.
    show : bool, default False
        If ``True``, every call to :py:meth:`project` pops up a window
        showing the projected points and their IDs.
    """

    def __init__(
        self,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        image_shape: Tuple[int, int] | None = None,
        show: bool = False,
        log: bool = False,
        ambif: bool = True, # Determines if exception is gonna be thrown when a point out of FOV
    ) -> None:
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)
        self.show = bool(show)
        self.dolog = bool(log)
        self.allMustBeInFOV = bool(ambif)

        # Canvas size for the live preview
        w = int(2 * self.cx) if image_shape is None else int(image_shape[0])
        h = int(2 * self.cy) if image_shape is None else int(image_shape[1])
        self._canvas_size = (w, h)  # (width, height)
        # Store for bounds checking and overlay
        self.image_shape = (w, h)
    
    def getK(self):
        return np.array([[self.fx, 0, self.cx],
                         [0, self.fy, self.cy],
                         [0,   0,   1]], float)

    def project(self, pts_cam: np.ndarray) -> np.ndarray:
        """Project 3‑D camera‑frame points with IDs to pixel coordinates with IDs.

        Parameters
        ----------
        pts_cam : array‑like, shape (N, 4) or (4,)
            Points in camera coordinates encoded as (id, X, Y, Z).

        Returns
        -------
        uv_id : ndarray, shape (N, 3)
            Projected pixel coordinates ``(id, u, v)``.
        """
        pts = np.asarray(pts_cam, dtype=np.float64)
        if pts.ndim == 1:
            pts = pts[None, :]
        if pts.shape[1] != 4:
            raise ValueError("Input must have shape (N,4) or (4,) with (id,x,y,z)")

        ids = pts[:, 0]
        X, Y, Z = pts[:, 1], pts[:, 2], pts[:, 3]
        if np.any(Z <= 0):
            raise ValueError("All Z values must be positive (in front of camera).")

        x = X / Z
        y = Y / Z

        u = self.fx * x + self.cx
        v = self.fy * y + self.cy
        uv_id = np.column_stack((ids, u, v))

        self._show_on_canvas(uv_id)

        return uv_id

    def _show_on_canvas(self, uv_id: np.ndarray) -> None:
        """Display projected pixels and IDs on a blank canvas via OpenCV."""
        width, height = self.image_shape
        canvas = np.zeros((height, width, 3), dtype=np.uint8)

        for pid, u, v in uv_id:
            u_int = int(round(u))
            v_int = int(round(v))
            if 0 <= u_int < width and 0 <= v_int < height:
                # draw point
                cv2.circle(canvas, (u_int, v_int), radius=3, color=(0, 255, 0), thickness=-1)
                # overlay ID
                cv2.putText(
                    canvas,
                    str(int(pid)),
                    (u_int + 5, v_int - 5),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=(255, 255, 255),
                    thickness=1,
                )
            else:
                if self.allMustBeInFOV:
                    raise ValueError(f"Point ID {int(pid)} at ({u_int}, {v_int}) is outside image bounds {self.image_shape}")
                else:
                    print(f"[WARNING] Point ID {int(pid)} at ({u_int}, {v_int}) is outside image bounds {self.image_shape}")

        if self.show:
            cv2.imshow("Pinhole Projection Preview", canvas)
            cv2.waitKey(1)

# ====================================================================== #
class SquaredPixelFocalCenteredPinholeCamera(PinholeCamera):
    """A *minimal* pinhole camera model with the following assumptions:

    * **Focal‑centred** – principal point sits exactly at the image centre.
    * **Square pixels** – pixel aspect ratio is 1 ⇒ `fx == fy == f`.
    * **No lens distortion**.

    Therefore the camera can be fully described by **only two inputs**:

    1. ``image_shape`` – the image dimensions in *pixels*.
    2. ``hfov_deg`` – the horizontal field‑of‑view in *degrees*.
    """

    def __init__(
        self,
        image_shape: Tuple[int, int],
        hfov_deg: float,
        show: bool = False,
        log: bool = False,
        ambif: bool = True, # Determines if exception is gonna be thrown when a point out of FOV
    ) -> None:
        W, H = int(image_shape[0]), int(image_shape[1])
        hfov_rad = math.radians(float(hfov_deg))
        if hfov_rad <= 0 or hfov_rad >= math.pi:
            raise ValueError("hfov_deg must be in the open interval (0, 180)")
        f = (W / 2.0) / math.tan(hfov_rad / 2.0)

        cx = W / 2.0
        cy = H / 2.0

        self.f = f
        self.hfov_deg = float(hfov_deg)
        self.vfov_deg = math.degrees(2.0 * math.atan((H / 2.0) / f))
        self.image_shape = (W, H)

        super().__init__(
            fx=f,
            fy=f,
            cx=cx,
            cy=cy,
            image_shape=(W, H),
            show=show,
            log=log,
            ambif=ambif
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(image_shape={self.image_shape}, "
            f"hfov_deg={self.hfov_deg:.2f}, f={self.f:.2f}px, "
            f"vfov_deg={self.vfov_deg:.2f})"
        )


SimpleCamera = SquaredPixelFocalCenteredPinholeCamera

__all__ = [
    "PinholeCamera",
    "SquaredPixelFocalCenteredPinholeCamera",
    "SimpleCamera",
]
