"""Procedurally drawn textures for details a coarse colour grid cannot carry
(Mario's eyes and cap emblem). Shapes come only from the kept 2-bit alpha
outline in the spec; everything inside is drawn here.
"""
import numpy as np

from cleanroom.gfx import strokefont

SCLERA = (246, 246, 250)
IRIS = (40, 110, 220)
PUPIL = (12, 12, 20)
SKIN = (254, 196, 140)
BROW = (60, 30, 12)
LASH = (40, 20, 10)

GAZE = {"center": (0, 0), "left": (-0.22, 0), "right": (0.22, 0), "up": (0, -0.22), "down": (0, 0.22)}


def _eye_boxes(alpha):
    """Per half of the texture: (brow mask, eye mask, bbox) from the outline."""
    h, w = alpha.shape
    out = []
    for x0, x1 in ((0, w // 2), (w // 2, w)):
        half = np.zeros_like(alpha, bool)
        half[:, x0:x1] = alpha[:, x0:x1] > 0
        rows = np.nonzero(half.any(1))[0]
        if not len(rows):
            continue
        # the brow is the first band of rows, separated from the eye by a gap
        gap = [r for r in range(rows[0], rows[-1]) if not half[r].any()]
        split = gap[0] if gap else rows[0] - 1
        brow = half.copy()
        brow[split:] = False
        eye = half.copy()
        eye[:split + 1] = False
        ys, xs = np.nonzero(eye)
        if len(ys):
            out.append((brow, eye, (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)))
    return out


def eyes(kind, alpha):
    h, w = alpha.shape
    img = np.zeros((h, w, 4), np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32) + 0.5
    for brow, eye, (bx0, by0, bx1, by1) in _eye_boxes(alpha):
        img[brow] = (*BROW, 255)
        img[eye] = (*SKIN, 255) if kind.startswith("closed") else (*SCLERA, 255)
        cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
        ew, eh = bx1 - bx0, by1 - by0
        if kind.startswith("closed"):
            lash = eye & (np.abs(yy - cy) < 0.9)
            img[lash] = (*LASH, 255)
            continue
        if kind == "dead":
            d1 = np.abs((xx - cx) / ew - (yy - cy) / eh) < 0.07
            d2 = np.abs((xx - cx) / ew + (yy - cy) / eh) < 0.07
            img[eye & (d1 | d2)] = (*PUPIL, 255)
            continue
        gx, gy = GAZE.get(kind, (0, 0))
        ix, iy = cx + gx * ew, cy + gy * eh
        r = np.hypot((xx - ix) / (ew * 0.31), (yy - iy) / (eh * 0.36))
        img[eye & (r < 1.0)] = (*IRIS, 255)
        img[eye & (r < 0.5)] = (*PUPIL, 255)
        hl = np.hypot(xx - (ix - ew * 0.1), yy - (iy - eh * 0.14)) < max(0.8, ew * 0.07)
        img[eye & hl] = (*SCLERA, 255)
        if kind == "half_closed":
            lid = eye & (yy < cy)
            img[lid] = (*SKIN, 255)
            img[eye & (np.abs(yy - cy) < 0.8)] = (*LASH, 255)
    return img


def emblem(alpha, fg=(230, 20, 30), bg=(250, 250, 250)):
    """White oval with a bold red M, inside the kept outline."""
    h, w = alpha.shape
    img = np.zeros((h, w, 4), np.float32)
    inside = alpha > 0
    img[inside] = (*bg, 255)
    ys, xs = np.nonzero(inside)
    if len(ys):
        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
        mw, mh = int((x1 - x0) * 0.62), int((y1 - y0) * 0.8)
        m = strokefont.render("M", mw, mh, thickness=max(1.0, mh * 0.12))
        ox, oy = x0 + ((x1 - x0) - mw) // 2, y0 + ((y1 - y0) - mh) // 2
        a = m[..., None]
        reg = img[oy:oy + mh, ox:ox + mw, :3]
        img[oy:oy + mh, ox:ox + mw, :3] = reg * (1 - a) + np.asarray(fg, np.float32) * a
    img[..., 3] = np.where(inside, alpha, 0)
    return img


def drawn(path, alpha):
    """Return an RGBA image for paths drawn here, else None."""
    name = path.rsplit("/", 1)[-1].split(".")[0]
    if name.startswith("mario_eyes_"):
        kind = name[len("mario_eyes_"):].replace("_unused", "")
        kind = "closed" if kind.startswith("closed") else kind
        return eyes(kind, alpha)
    if name in ("mario_logo", "mario_cap_logo"):
        return emblem(alpha)
    return None
