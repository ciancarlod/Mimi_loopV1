#!/usr/bin/env python3
"""v41t-ex: post-pass on frozen v41r-ex.

Fixes the v41s-ex face ghost (frost plate leaked onto the cat in the
right pane). Rebuilds from frozen v41r-ex lossless.

Frost stays on the left-pane snow bank only. Lamp flicker is retuned so
the desk pool and lamp-side wall move about 4-5%, with a softer bulb.
Tail flicks are a bending chain (base angle, mid bend, tip whip), not a
translated sprite; tip travel is ~35-45 px quick / ~60-70 px slow; rest
frames are near-still. Inverse-spine warp avoids horizontal streaks.
n432 muzzle patch is applied only around that meow.

Does not edit or overwrite earlier named encodes or builders.
Audio is the approved v41 PCM (mux only). Title is copied through.
241-periodic. 1280x720 24fps x264 slow CRF12.
"""
from __future__ import annotations

import hashlib
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

P = Path('/workspace/lofi-v41-meow')
sys.path.insert(0, str(P))
import build_v41h as Hh
import build_v41g as G
import build_v41f_r2 as Rf

VER = 'v41t-ex'
WORK = P / VER
QC = P / 'qc' / VER
SRC = P / 'v41r-ex' / 'v41r-ex-span-890-915-lossless.mkv'
SPRITE_FIXED = P / 'v41o-ex-pre' / 'tail-sprite-fixed-v41o-ex-pre.png'
MATTE = Hh.MATTE
PCM = Hh.PCM

V41R_SHA = 'b216388508c7b48e12866f135ba30971097dd3af10f81933e95c8dd9baaf8e1a'
V41R_LOS_SHA = '37bef6b14b1b74e79c24b14c74f4d565138bc0216d0d0e788addbaa13fa004b9'

NFR, W, H, NCYCLE = 600, 1280, 720, 241
TITLE = Hh.TITLE
Yg, Xg = np.mgrid[0:H, 0:W].astype(np.float32)
BULB = np.array([315.0, 168.0], np.float32)

PANE_L = np.array([[266, 0], [558, 0], [556, 400], [268, 400]], np.int32)
PANE_R = np.array([[600, 0], [840, 0], [838, 400], [600, 400]], np.int32)

# Flick schedule in phase (241-periodic, rest at ph 0 / 241)
# (start, t_out, t_in, mid_px, tip_px, overshoot)
QUICK_A = (18.0, 8.0, 12.0, 10.0, 18.0, 0.16)
SLOW_F = (75.0, 16.0, 20.0, 18.0, 32.0, 0.12)
QUICK_B = (155.0, 8.0, 12.0, 10.0, 18.0, 0.16)
MID_LAG, TIP_LAG = 2.2, 5.0


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for ch in iter(lambda: f.read(1 << 20), b''):
            h.update(ch)
    return h.hexdigest()


def luma(im):
    im = im.astype(np.float32)
    return 0.114 * im[..., 0] + 0.587 * im[..., 1] + 0.299 * im[..., 2]


def sstep(x):
    x = np.clip(x, 0, 1)
    return x * x * x * (x * (6 * x - 15) + 10)


def norm_alpha(a):
    a = a.astype(np.float32)
    if a.max() > 1.5:
        a = a / 255.0
    return np.clip(a, 0, 1)


def pane_masks():
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [PANE_L, PANE_R], 1)
    return m


def left_pane_mask():
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [PANE_L], 1)
    return m


# ---------------- tail deform: segment chain, follow-through ----------------
def flick_env(t, t_out, t_in, overshoot):
    """0..1 out, then settle with a slight overshoot. Works on arrays."""
    t = np.asarray(t, np.float32)
    out = np.zeros_like(t, np.float32)
    m1 = (t >= 0) & (t <= t_out)
    u = np.zeros_like(t)
    u[m1] = t[m1] / max(t_out, 1e-6)
    out[m1] = 0.5 - 0.5 * np.cos(np.pi * u[m1])
    m2 = (t > t_out) & (t <= t_out + t_in)
    u[m2] = (t[m2] - t_out) / max(t_in, 1e-6)
    uu = u[m2]
    out[m2] = (np.exp(-2.1 * uu) * np.cos(np.pi * uu)
               - overshoot * np.sin(np.pi * uu) * np.exp(-1.6 * uu))
    return out


# Angle amplitudes (degrees) for (quick, slow) at base / mid / tip.
# Tuned so tip travel is ~35-45 px on the quick flicks and ~60-70 px
# on the slow flick, with a visible curve (base first, mid bends, tip whips).
BASE_DEG = (6.5, 10.2)
MID_DEG = (8.5, 14.5)
TIP_DEG = (11.0, 18.0)


def resample_poly(poly, n=24):
    arc, tot = Hh._arc(poly)
    s = np.linspace(0.0, tot, n)
    return np.stack([
        np.interp(s, arc, poly[:, 0]),
        np.interp(s, arc, poly[:, 1]),
    ], 1).astype(np.float32)


def _flick_deg(ph, lag, qdeg, sdeg):
    q = flick_env(ph - QUICK_A[0] - lag, QUICK_A[1], QUICK_A[2], QUICK_A[5])
    sl = flick_env(ph - SLOW_F[0] - lag, SLOW_F[1], SLOW_F[2], SLOW_F[5])
    qb = flick_env(ph - QUICK_B[0] - lag, QUICK_B[1], QUICK_B[2], QUICK_B[5])
    return (q + qb) * qdeg + sl * sdeg


def deform_flicks(rest, ph):
    """Bend the rest chain in place. Root pinned; no whole-sprite translation.

    Base angle changes first, mid segments add curvature, the tip whips
    later with a little overshoot. Segment lengths are preserved.
    """
    dxy = np.diff(rest, axis=0)
    seglen = np.hypot(dxy[:, 0], dxy[:, 1])
    rest_ang = np.arctan2(dxy[:, 1], dxy[:, 0])
    arc = np.concatenate([[0.0], np.cumsum(seglen)])
    s = arc / (arc[-1] + 1e-8)
    pts = np.zeros_like(rest)
    pts[0] = rest[0]
    for i in range(len(seglen)):
        sm = 0.5 * (s[i] + s[i + 1])
        pb = _flick_deg(ph, 0.0, BASE_DEG[0], BASE_DEG[1])
        pm = _flick_deg(ph, MID_LAG, MID_DEG[0], MID_DEG[1]) * sstep((sm - 0.12) / 0.50)
        pt = _flick_deg(ph, TIP_LAG, TIP_DEG[0], TIP_DEG[1]) * (sm ** 1.65)
        ang = rest_ang[i] + np.radians(pb + pm + pt)
        pts[i + 1] = pts[i] + seglen[i] * np.array([np.cos(ang), np.sin(ang)], np.float32)
    return pts


def prep_tail_padded(sprite_path, PAD=90):
    """Same padded prep as v41o-ex, so the hook tip is not clipped."""
    sp = cv2.imread(str(sprite_path), cv2.IMREAD_UNCHANGED)
    assert sp is not None and sp.shape[2] == 4, sprite_path
    sp = cv2.copyMakeBorder(sp, PAD, PAD, PAD, PAD, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
    a = sp[:, :, 3].astype(np.float32) / 255.0
    rgb = sp[:, :, :3].astype(np.float32) * 0.90
    poly_s = np.array([
        [25, 305], [40, 255], [55, 210], [75, 170], [110, 130],
        [155, 100], [200, 78], [245, 68], [285, 78], [312, 108],
    ], np.float32) + PAD
    base = poly_s[0].copy()
    tip = poly_s[-1].copy()
    sh, sw = sp.shape[:2]
    nh, nw = int(round(sh * Hh.TAIL_SCALE)), int(round(sw * Hh.TAIL_SCALE))
    rgb_s = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    a_s = cv2.resize(a, (nw, nh), interpolation=cv2.INTER_AREA)
    poly_s = poly_s * Hh.TAIL_SCALE
    base_s = base * Hh.TAIL_SCALE
    tip_s = tip * Hh.TAIL_SCALE
    v_src = tip_s - base_s
    v_dst = Hh.TIP_W - Hh.BASE_W
    ang = np.degrees(np.arctan2(v_dst[1], v_dst[0]) - np.arctan2(v_src[1], v_src[0]))
    M = cv2.getRotationMatrix2D(tuple(map(float, base_s)), -ang, 1.0)
    rgb_r = cv2.warpAffine(rgb_s, M, (nw, nh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    a_r = cv2.warpAffine(a_s, M, (nw, nh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    ones = np.ones((len(poly_s), 1), np.float32)
    pr = (M @ np.hstack([poly_s, ones]).T).T
    br = (M @ np.array([base_s[0], base_s[1], 1.0], np.float32)).ravel()
    ox, oy = Hh.BASE_W[0] - br[0], Hh.BASE_W[1] - br[1]
    rest_poly = pr + np.array([ox, oy], np.float32)
    dt = cv2.distanceTransform((a_r > 0.25).astype(np.uint8), cv2.DIST_L2, 3)
    widths = []
    for q in pr:
        x, y = int(np.clip(round(q[0]), 0, nw - 1)), int(np.clip(round(q[1]), 0, nh - 1))
        widths.append(float(dt[y, x]))
    widths = np.maximum(np.array(widths, np.float32), 8.0)
    world_rgb = np.zeros((H, W, 3), np.float32)
    world_a = np.zeros((H, W), np.float32)

    def _paste(dst, src, x0, y0):
        h, w = src.shape[:2]
        y1, x1 = y0 + h, x0 + w
        sy0 = sx0 = 0
        sy1, sx1 = h, w
        if y0 < 0:
            sy0 = -y0
            y0 = 0
        if x0 < 0:
            sx0 = -x0
            x0 = 0
        if y1 > dst.shape[0]:
            sy1 -= (y1 - dst.shape[0])
            y1 = dst.shape[0]
        if x1 > dst.shape[1]:
            sx1 -= (x1 - dst.shape[1])
            x1 = dst.shape[1]
        if y1 > y0 and x1 > x0:
            dst[y0:y1, x0:x1] = src[sy0:sy1, sx0:sx1]

    _paste(world_rgb, rgb_r, int(round(ox)), int(round(oy)))
    _paste(world_a, a_r, int(round(ox)), int(round(oy)))
    rest_hi = resample_poly(rest_poly, 24)
    arc0, _ = Hh._arc(rest_poly)
    arc1, _ = Hh._arc(rest_hi)
    widths_hi = np.interp(arc1, arc0, widths).astype(np.float32)
    return {
        'rgb': rgb_r, 'a': a_r, 'ox': ox, 'oy': oy,
        'rest': rest_hi, 'nw': nw, 'nh': nh,
        'widths': widths_hi,
        'world_rgb': world_rgb, 'world_a': world_a,
    }


def build_tail_bg(src, mattes, tail):
    """Median chair/plant behind the tail, from frames where the hook isn't there."""
    gate = ((Yg > 500) & (Xg > 900)).astype(np.float32)
    occ = np.zeros((H, W), np.float32)
    acc = np.zeros((H, W, 3), np.float64)
    wsum = np.zeros((H, W), np.float64)
    wa0 = tail['world_a']
    # occupancy of rest + a generous dilation (covers the wag)
    occ = cv2.dilate((wa0 > 0.08).astype(np.uint8),
                     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))).astype(np.float32)
    occ = occ * gate
    for i in range(0, NFR, 1):
        if i >= 241:
            break
        a = norm_alpha(mattes[i]) * gate
        free = (1.0 - sstep((a - 0.12) / 0.20)) * occ
        if free.max() < 0.05:
            continue
        acc += src[i].astype(np.float64) * free[..., None]
        wsum += free
    bg = src[121].astype(np.float32).copy()  # calm / plant-rest frame
    have = wsum > 8.0
    bg[have] = (acc[have] / wsum[have, None]).astype(np.float32)
    return np.clip(bg, 0, 255), np.clip(cv2.GaussianBlur(occ, (0, 0), 3.0), 0, 1)


# ---------------- plant: mainly the slow flick, per-leaf delay ----------------
def plant_delay_map(plate):
    solid = (plate > 0.25).astype(np.uint8)
    n, lab, stats, cents = cv2.connectedComponentsWithStats(solid, 8)
    dmap = np.zeros((H, W), np.float32)
    contact = G.CONTACT
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 40:
            continue
        cx, cy = cents[i]
        dist = float(np.hypot(cx - contact[0], cy - contact[1]))
        delay = 1.5 + 7.5 * float(np.clip(dist / 220.0, 0, 1))
        dmap[lab == i] = delay
    dmap = cv2.GaussianBlur(dmap, (0, 0), 3.0)
    return dmap


def plant_drive(ph, delay):
    """Scalar or array: mostly the slow flick, a little of the quick ones."""
    sig = 0.90 * flick_env(ph - SLOW_F[0] - 4.0 - delay, SLOW_F[1], SLOW_F[2], 0.10)
    sig = sig + 0.12 * flick_env(ph - QUICK_A[0] - 2.0 - delay, QUICK_A[1], QUICK_A[2], 0.08)
    sig = sig + 0.12 * flick_env(ph - QUICK_B[0] - 2.0 - delay, QUICK_B[1], QUICK_B[2], 0.08)
    return sig


def apply_plant_flicks(img, plate, cat_a, ph, dmap, rest_plant):
    # restore rest plant, then warp
    live = np.clip(plate, 0, 1)
    live = live * (1.0 - sstep((norm_alpha(cat_a) - 0.25) / 0.25))
    live_s = cv2.GaussianBlur(live, (0, 0), 1.0)
    if live_s.max() < 0.05:
        return img
    imgf = img.astype(np.float32)
    imgf = imgf * (1.0 - live_s[..., None]) + rest_plant * live_s[..., None]
    A = plant_drive(ph, dmap) * live_s
    if float(np.max(np.abs(A))) < 0.03:
        return np.clip(imgf, 0, 255)
    # ~3.5° at the peak on the nearest leaves, plus a short push
    ang = np.radians(3.6) * A
    trans = 5.5 * A
    PUSH = np.array([0.92, 0.39], np.float32)
    PUSH = PUSH / np.hypot(*PUSH)
    stem = G.STEM
    vx = Xg - stem[0]
    vy = Yg - stem[1]
    ca, sa = np.cos(ang), np.sin(ang)
    sx = stem[0] + ca * vx + sa * vy - (trans * PUSH[0])
    sy = stem[1] - sa * vx + ca * vy - (trans * PUSH[1])
    y0, y1, x0, x1 = 450, H, 1000, W
    mapx = Xg.copy()
    mapy = Yg.copy()
    mapx[y0:y1, x0:x1] = sx[y0:y1, x0:x1]
    mapy[y0:y1, x0:x1] = sy[y0:y1, x0:x1]
    warped = cv2.remap(imgf, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    wa = cv2.remap(live_s, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    wa = np.clip(wa, 0, 1)
    out = imgf * (1.0 - wa[..., None]) + warped * wa[..., None]
    return np.clip(out, 0, 255)


# ---------------- frost: static plate, flakes in front ----------------
def build_frost(src, cat_union):
    """Static left-bank plate only. Never the right pane, the cat, or the plant.

    The v41s-ex bug was `pane` = both panes plus Xg>=330, so cream fur in
    PANE_R was treated as frost and the plate (src[0] on the face) ghosted
    a second face onto blinks/meows/head-turns.
    """
    pane_l = left_pane_mask()
    y0, y1, x0, x1 = 220, 410, 250, 560
    stack = np.stack([src[i][y0:y1, x0:x1] for i in range(241)], 0).astype(np.float32)
    med = np.median(stack, axis=0)
    plate = src[0].astype(np.float32).copy()
    plate[y0:y1, x0:x1] = med
    Y = luma(plate)
    B, Gc, R = plate[..., 0], plate[..., 1], plate[..., 2]
    chroma = np.maximum(np.maximum(R, Gc), B) - np.minimum(np.minimum(R, Gc), B)
    lower = (Yg >= 255) & (Yg <= 400)
    leaf = (Gc > R + 4) & (Gc > B + 0)
    catx = sstep((cat_union.astype(np.float32) - 0.04) / 0.12)
    raw = ((pane_l > 0) & lower & (Y > 42) & (Y < 200) & (chroma < 85)
           & (Xg >= 330) & (Xg <= 556) & (leaf == 0) & (catx < 0.15)).astype(np.uint8)
    raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    nlab, lab, stats, _ = cv2.connectedComponentsWithStats(raw, 8)
    keep = np.zeros_like(raw)
    for i in range(1, nlab):
        if int(stats[i, cv2.CC_STAT_AREA]) >= 250:
            keep[lab == i] = 1
    m = keep.astype(np.float32)
    rows = np.arange(H)[:, None]
    top = np.where(keep > 0, rows, H).min(axis=0)
    dist_top = (top[None, :] - rows).astype(np.float32)
    crest = sstep((28.0 - np.abs(dist_top)) / 28.0) * (Xg >= 330) * (Xg <= 550) * (pane_l > 0)
    m = np.clip(cv2.GaussianBlur(m, (0, 0), 2.2) + 0.35 * crest * (keep == 0) * (Yg < 400), 0, 1)
    m = m * (pane_l > 0).astype(np.float32) * (leaf == 0).astype(np.float32)
    m = cv2.GaussianBlur(m, (0, 0), 1.4)
    m = m * sstep((Xg - 328.0) / 10.0) * sstep((556.0 - Xg) / 8.0)
    m = m * (1.0 - catx)
    return plate, np.clip(m, 0, 1)


def flake_front(img, plate, pane):
    """Compact bright residuals vs the static plate — falling flakes."""
    Y = luma(img)
    Yp = luma(plate)
    res = Y - Yp
    B, Gc, R = img[..., 0].astype(np.float32), img[..., 1].astype(np.float32), img[..., 2].astype(np.float32)
    chroma = np.maximum(np.maximum(R, Gc), B) - np.minimum(np.minimum(R, Gc), B)
    # flakes only in the left pane; never on the cat or the right pane
    pane_l = left_pane_mask()
    cand = ((pane_l > 0) & (res > 7) & (Y > 55) & (chroma < 90)
            & (Xg <= 556)).astype(np.uint8)
    nlab, lab, stats, _ = cv2.connectedComponentsWithStats(cand, 8)
    keep = np.zeros_like(cand)
    for i in range(1, nlab):
        area = int(stats[i, cv2.CC_STAT_AREA])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        if 2 <= area <= 90 and bw <= 18 and bh <= 18:
            keep[lab == i] = 1
    if not keep.any():
        return np.zeros((H, W), np.float32)
    keep = cv2.dilate(keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    return np.clip(cv2.GaussianBlur(keep.astype(np.float32), (0, 0), 0.7), 0, 1)


def apply_frost(img, plate, frost_m, pane):
    imgf = img.astype(np.float32)
    out = imgf * (1.0 - frost_m[..., None]) + plate * frost_m[..., None]
    fl = flake_front(imgf, plate, pane)
    # flakes in front of the plate (and in the open pane)
    out = out * (1.0 - fl[..., None]) + imgf * fl[..., None]
    return np.clip(out, 0, 255)


# ---------------- lamp flicker: whole lamp-lit scene, window constant -----
def v28_pulse(ph):
    # Gentle 3-cycle raised cosine (the v28 **2 pulse is too peaky / strobe-like).
    return float(0.5 - 0.5 * np.cos(2 * np.pi * 3 * ph / NCYCLE))


def flicker_signal(ph):
    pulse = v28_pulse(ph)
    slow = (pulse - 0.5) * 2.0  # -1 .. 1
    dips = (0.50 * np.sin(2 * np.pi * 7 * ph / NCYCLE + 0.3)
            + 0.28 * np.sin(2 * np.pi * 13 * ph / NCYCLE + 1.1)
            + 0.14 * np.sin(2 * np.pi * 19 * ph / NCYCLE + 2.0)
            + 0.08 * np.sin(2 * np.pi * 29 * ph / NCYCLE + 0.7))
    return float(np.clip(0.78 * slow + 0.22 * dips, -1.1, 1.1))


def lamp_weights(pane, cat_union=None):
    """Separate room (desk/wall/face) and bulb weights.

    Open glass stays constant; the fixture and cat-in-front-of-pane are
    included. Room weight is lifted on the desk pool and lamp-side wall
    so those regions can hit ~4-5% p2p without a strobing bulb.
    """
    d = np.hypot(Xg - BULB[0], Yg - BULB[1])
    fall = np.exp(-0.5 * (d / 320.0) ** 2)
    desk = np.exp(-0.5 * (((Xg - 280.0) / 200.0) ** 2 + ((Yg - 620.0) / 150.0) ** 2))
    wall = np.exp(-0.5 * (((Xg - 130.0) / 160.0) ** 2 + ((Yg - 250.0) / 200.0) ** 2))
    lamp = Rf.lamp_masks()
    protect = lamp['protect'].astype(np.float32)
    glass = pane.astype(np.float32) * (1.0 - sstep((protect - 0.08) / 0.15))
    face = np.zeros((H, W), np.float32)
    if cat_union is not None:
        glass = glass * (1.0 - sstep((cat_union.astype(np.float32) - 0.08) / 0.15))
        face = np.exp(-0.5 * (((Xg - 640.0) / 100.0) ** 2 + ((Yg - 300.0) / 90.0) ** 2))
        face = face * cat_union
    w_room = np.clip(0.28 * fall + 1.00 * desk + 0.95 * wall + 0.50 * face, 0, 1)
    w_room = w_room * (1.0 - glass)
    # do not pile room falloff onto the fixture (that made the bulb swing
    # larger than the desk/wall target)
    w_room = w_room * (1.0 - 0.85 * sstep((protect - 0.12) / 0.25))
    w_bulb = np.clip(protect, 0, 1) * (1.0 - glass)
    y0, y1, x0, x1 = TITLE
    w_room[y0:y1, x0:x1] = 0
    w_bulb[y0:y1, x0:x1] = 0
    w_room = np.clip(cv2.GaussianBlur(w_room, (0, 0), 5.0), 0, 1)
    w_bulb = np.clip(cv2.GaussianBlur(w_bulb, (0, 0), 3.0), 0, 1)
    return w_room, w_bulb


def apply_flicker(img, w_room, w_bulb, ph):
    sig = flicker_signal(ph)
    # ~4-5% p2p on desk/wall when w_room ~ 1 and sig spans about +/-1
    # softer bulb swing (~2%) so the fixture does not strobe
    warm = np.array([0.72, 0.92, 1.06], np.float32)
    fk = (0.022 * sig * w_room + 0.013 * sig * w_bulb)
    out = img.astype(np.float32) * (1.0 + fk[..., None] * warm)
    return np.clip(out, 0, 255)


# ---------------- muzzle blotch ----------------
def fix_muzzle(img, src_closed=None):
    """Blend the pink muzzle blotch toward same-frame cream; never touch the opening."""
    imgf = img.astype(np.float32)
    y0, y1, x0, x1 = 368, 450, 730, 815
    sub = imgf[y0:y1, x0:x1]
    B, G, R = sub[..., 0], sub[..., 1], sub[..., 2]
    Y = luma(sub)
    opening = ((Y < 115) & (R > G + 6) & (R > B + 4) & (Y > 20)).astype(np.uint8)
    opening = cv2.dilate(opening, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    cream = (Y > 88) & ((R - B) < 48) & (opening == 0) & (np.abs(R - G) < 35)
    if int(cream.sum()) < 25:
        return np.clip(imgf, 0, 255)
    ref = np.median(sub[cream], axis=0)
    blot = ((R > ref[2] + 8) & ((R - B) > (ref[2] - ref[0]) + 6)
            & (opening == 0) & (Y > 55) & (Y < 200))
    if not blot.any():
        return np.clip(imgf, 0, 255)
    m = cv2.GaussianBlur(blot.astype(np.float32), (0, 0), 1.8)
    m = np.clip(m * 1.10, 0, 1) * (1.0 - opening.astype(np.float32))
    fill = cv2.inpaint(np.clip(sub, 0, 255).astype(np.uint8),
                       (m > 0.2).astype(np.uint8) * 255, 7, cv2.INPAINT_TELEA).astype(np.float32)
    # mix inpaint with the local cream so it stays white/cream, not grey
    fill = 0.55 * fill + 0.45 * ref
    sub2 = sub * (1.0 - m[..., None]) + fill * m[..., None]
    out = imgf.copy()
    out[y0:y1, x0:x1] = sub2
    return np.clip(out, 0, 255)


def apply_tail(img, cat_a, ph, tail, mic, bg, occ):
    """Inverse-map the painted sprite along the bent spine.

    Sample by closest point on the *deformed* spine, then the matching
    rest cross-section. This bends the tail instead of sliding it, and
    avoids the dark horizontal slice-seams of rest-space translation.
    """
    imgf = img.astype(np.float32)
    occ_s = occ * sstep((Yg - 500.0) / 16.0) * sstep((Xg - 900.0) / 16.0)
    cleared = imgf * (1.0 - occ_s[..., None]) + bg * occ_s[..., None]

    rest = tail['rest']
    defp = deform_flicks(rest, ph)
    arc_r, tot_r = Hh._arc(rest)
    arc_d, tot_d = Hh._arc(defp)
    pad = 90
    xs = np.concatenate([rest[:, 0], defp[:, 0]])
    ys = np.concatenate([rest[:, 1], defp[:, 1]])
    x0, x1 = int(max(0, xs.min() - pad)), int(min(W, xs.max() + pad + 1))
    y0, y1 = int(max(0, ys.min() - pad)), int(min(H, ys.max() + pad + 1))
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    s, sd, _, _ = Hh._closest(xx, yy, defp, arc_d, tot_d)
    rx = np.interp(s * tot_r, arc_r, rest[:, 0])
    ry = np.interp(s * tot_r, arc_r, rest[:, 1])
    tg = np.zeros_like(rest)
    tg[1:-1] = rest[2:] - rest[:-2]
    tg[0] = rest[1] - rest[0]
    tg[-1] = rest[-1] - rest[-2]
    ln = np.sqrt((tg ** 2).sum(1)) + 1e-8
    tg = tg / ln[:, None]
    nrm = np.stack([-tg[:, 1], tg[:, 0]], 1)
    nx = np.interp(s * tot_r, arc_r, nrm[:, 0])
    ny = np.interp(s * tot_r, arc_r, nrm[:, 1])
    mapx = (rx + sd * nx).astype(np.float32)
    mapy = (ry + sd * ny).astype(np.float32)
    rgb = cv2.remap(tail['world_rgb'], mapx, mapy, cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT)
    al = cv2.remap(tail['world_a'], mapx, mapy, cv2.INTER_LINEAR,
                   borderMode=cv2.BORDER_CONSTANT)
    w_s = np.interp(np.clip(s, 0, 1) * tot_r, arc_r, tail['widths'])
    al = np.clip(al * sstep((w_s + 28.0 - np.abs(sd)) / 6.0), 0, 1)
    la_full = np.zeros((H, W), np.float32)
    layer = np.zeros((H, W, 3), np.float32)
    la_full[y0:y1, x0:x1] = al
    layer[y0:y1, x0:x1] = rgb
    sh = cv2.GaussianBlur(la_full, (0, 0), 8.0)
    base_w = np.exp(-0.5 * ((Xg - Hh.BASE_W[0]) ** 2 + (Yg - Hh.BASE_W[1]) ** 2) / (48.0 ** 2))
    shadow = 0.35 * sh * base_w * (1.0 - la_full) * (1.0 - np.clip(cat_a, 0, 1))
    comp = cleared * (1.0 - 0.75 * shadow[..., None])
    comp = comp * (1.0 - la_full[..., None]) + layer * la_full[..., None]
    cat_s = cv2.GaussianBlur(np.clip(cat_a, 0, 1), (0, 0), 0.8)
    protect = np.clip(np.maximum(cat_s, mic), 0, 1)
    comp = imgf * protect[..., None] + comp * (1.0 - protect[..., None])
    # pin the root under the haunch using the incoming body, not the cleared bg
    root = np.exp(-0.5 * ((Xg - Hh.BASE_W[0]) ** 2 + (Yg - Hh.BASE_W[1]) ** 2) / (56.0 ** 2))
    overlap = np.clip(np.clip(cat_a, 0, 1) * (0.25 + 0.80 * root) * sstep((Yg - 500.0) / 20.0), 0, 1)
    comp = comp * (1.0 - overlap[..., None]) + imgf * overlap[..., None]
    return np.clip(comp, 0, 255), defp[-1]


def process_one(src, cat_a, ph, ctx, i):
    img = src
    # muzzle patch only around the big n432 meow — never on other faces
    if 418 <= i <= 448:
        img = fix_muzzle(img)
    img, _ = apply_tail(img, cat_a, ph, ctx['tail'], ctx['mic'], ctx['tail_bg'], ctx['occ'])
    img = apply_plant_flicks(img, ctx['plate'], cat_a, ph, ctx['pdelay'], ctx['rest_plant'])
    img = apply_frost(img, ctx['frost_plate'], ctx['frost_m'], ctx['pane'])
    img = apply_flicker(img, ctx['w_room'], ctx['w_bulb'], ph)
    img = np.clip(img, 0, 255).astype(np.uint8)
    y0, y1, x0, x1 = TITLE
    img = img.copy()
    img[y0:y1, x0:x1] = src[y0:y1, x0:x1]
    return img



def face_compare_grid(out, src, dest):
    """Face crop every 10 frames over one cycle, t-ex vs r-ex, plus abs-diff."""
    dest.mkdir(parents=True, exist_ok=True)
    y0, y1, x0, x1 = 160, 460, 570, 950
    tiles_t, tiles_r, tiles_d = [], [], []
    report = []
    for i in range(0, 241, 10):
        if out[i] is None:
            continue
        tface = out[i][y0:y1, x0:x1]
        rface = src[i][y0:y1, x0:x1]
        # residual after undoing a uniform lamp gain (median ratio on mid face)
        tf = tface.astype(np.float32)
        rf = rface.astype(np.float32)
        # mid-face, avoid edges
        mid = tf[80:220, 80:300]
        midr = rf[80:220, 80:300]
        ratio = np.median((mid + 1.0) / (midr + 1.0))
        adj = np.clip(rf * ratio, 0, 255)
        mad = np.mean(np.abs(tf - adj))
        p95 = np.percentile(np.abs(tf - adj), 95)
        mx = np.max(np.abs(tf - adj))
        report.append((i, ratio, mad, p95, mx))
        def lab(im, tag):
            z = cv2.resize(np.clip(im, 0, 255).astype(np.uint8), (190, 150),
                           interpolation=cv2.INTER_AREA)
            cv2.putText(z, tag, (6, 18), cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 255, 255), 1)
            return z
        tiles_t.append(lab(tface, f't n{i}'))
        tiles_r.append(lab(rface, f'r n{i}'))
        diff = np.clip(np.abs(tf - adj) * 6.0, 0, 255)
        tiles_d.append(lab(diff, f'd n{i}'))
        # also write 2x crops for the required look-at-every-tile step
        cv2.imwrite(str(dest / f'face-t-2x-n{i}.png'),
                    cv2.resize(tface, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST))
        cv2.imwrite(str(dest / f'face-r-2x-n{i}.png'),
                    cv2.resize(rface, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST))
    def sheet(cells, name, cols=5):
        if not cells:
            return
        while len(cells) % cols:
            cells.append(np.zeros_like(cells[0]))
        rows = [np.hstack(cells[i:i + cols]) for i in range(0, len(cells), cols)]
        path = dest / name
        cv2.imwrite(str(path), np.vstack(rows))
        print('face grid', path)
    sheet(tiles_t, 'face-grid-t.png')
    sheet(tiles_r, 'face-grid-r.png')
    sheet(tiles_d, 'face-grid-diff.png')
    # paired r|t rows
    paired = []
    for a, b in zip(tiles_r, tiles_t):
        paired.append(np.vstack([a, b]))
    sheet(paired, 'face-grid-r-over-t.png', cols=5)
    print('face vs r-ex (lamp-gain adjusted) every 10 frames:')
    for i, ratio, mad, p95, mx in report:
        print(f'  n{i:3d} gain={ratio:.4f} mad={mad:.2f} p95={p95:.2f} max={mx:.1f}')
    return report


def mouth_qc(out, src, dest):
    dest.mkdir(parents=True, exist_ok=True)
    for i in (138, 432):
        if out[i] is None:
            continue
        pair = np.hstack([src[i][360:460, 640:820], out[i][360:460, 640:820]])
        z = cv2.resize(pair, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(dest / f'mouth-r-vs-t-n{i}.png'), z)
        print('mouth qc', dest / f'mouth-r-vs-t-n{i}.png')

def process(src, mattes, ctx, preview=None):
    want = set(preview) if preview else set(range(NFR))
    out = [None] * NFR
    for i in range(NFR):
        if i not in want:
            continue
        ph = Hh.phase_of(i)
        out[i] = process_one(src[i], norm_alpha(mattes[i]), ph, ctx, i)
        if preview or i % 25 == 0:
            Y = luma(out[i])
            print(f'  n{i} ph={ph} glob={Y.mean():.1f} bulb={Y[155:185, 300:330].mean():.1f}',
                  flush=True)
    return out


def plot_lamp(out, src, dest):
    dest.mkdir(parents=True, exist_ok=True)
    regs = {
        'bulb': (155, 185, 300, 330),
        'desk': (560, 700, 180, 420),
        'cheek': (250, 310, 600, 700),
        'wall': (80, 200, 80, 180),
        'win': (80, 200, 400, 540),
    }
    colors = {
        'bulb': (0, 200, 255), 'desk': (40, 90, 200), 'cheek': (40, 80, 220),
        'wall': (180, 100, 40), 'win': (80, 80, 200),
    }
    curves = {}
    for name, (y0, y1, x0, x1) in regs.items():
        ys = np.zeros(241, np.float32)
        for i in range(241):
            im = out[i] if (out[i] is not None) else src[i]
            ys[Hh.phase_of(i)] = float(luma(im[y0:y1, x0:x1]).mean())
        curves[name] = ys
    b = curves['bulb']
    print('lamp p2p / corr vs bulb (first cycle of this output):')
    for name, v in curves.items():
        pct = 100 * (v.max() - v.min()) / max(v.mean(), 1)
        c = 1.0 if name == 'bulb' else float(np.corrcoef(b, v)[0, 1])
        print(f'  {name} mean={v.mean():.2f} p2p={v.max()-v.min():.2f} ({pct:.2f}%) corr={c:.3f}')
        np.save(dest / f'luma-{name}.npy', v)
    # z-score plot
    Wd, Hd = 1200, 480
    padl, padr, padt, padb = 70, 20, 36, 40
    pw, phg = Wd - padl - padr, Hd - padt - padb
    canvas = np.full((Hd, Wd, 3), 245, np.uint8)
    cv2.rectangle(canvas, (padl, padt), (Wd - padr, Hd - padb), (255, 255, 255), -1)
    for k in range(5):
        y = padt + int(k * phg / 4)
        cv2.line(canvas, (padl, y), (Wd - padr, y), (220, 220, 220), 1)
    for name, v in curves.items():
        z = (v - v.mean()) / max(v.std(), 1e-6)
        z = np.clip(z, -3, 3)
        pts = []
        for i, zi in enumerate(z):
            x = padl + int(i * (pw - 1) / 240)
            y = padt + int((3 - zi) / 6 * phg)
            pts.append((x, y))
        cv2.polylines(canvas, [np.array(pts, np.int32)], False, colors[name], 2, cv2.LINE_AA)
    x = padl + 8
    for name, col in colors.items():
        cv2.putText(canvas, name, (x, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
        x += 90
    cv2.imwrite(str(dest / 'lamp-luma.png'), canvas)
    # brightest / dimmest bulb frames
    imax, imin = int(b.argmax()), int(b.argmin())
    print(f'  bulb brightest n{imax} Y={b[imax]:.2f}  dimmest n{imin} Y={b[imin]:.2f}')
    return imax, imin


def tail_strip(out, dest, every=4):
    cells = []
    for i in range(0, 241, every):
        if out[i] is None:
            continue
        c = out[i][500:720, 900:1280].copy()
        c = np.clip(c.astype(np.float32) * 1.45 + 18, 0, 255).astype(np.uint8)
        cv2.putText(c, f'n{i}', (6, 18), cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 255, 255), 1)
        cells.append(cv2.resize(c, (190, 110)))
    if not cells:
        return
    while len(cells) % 8:
        cells.append(np.zeros_like(cells[0]))
    rows = [np.hstack(cells[i:i + 8]) for i in range(0, len(cells), 8)]
    path = dest / 'tail-strip.png'
    cv2.imwrite(str(path), np.vstack(rows))
    print('tail strip', path)


def save_preview(out, src, idx, dest, bright=None, dim=None):
    dest.mkdir(parents=True, exist_ok=True)
    crops = {
        'head': (160, 460, 570, 950),
        'mouth': (360, 460, 640, 820),
        'muzzle': (360, 450, 700, 820),
        'steam': (250, 520, 300, 700),
        'lamp': (80, 220, 200, 460),
        'frost': (230, 410, 250, 560),
        'tail': (500, 720, 900, 1280),
        'plant': (460, 720, 1000, 1280),
        'screen': (360, 580, 0, 230),
    }
    thumbs = []
    for i in idx:
        if out[i] is None:
            continue
        cv2.imwrite(str(dest / f'full-n{i}.png'), out[i])
        for tag, (ya, yb, xa, xb) in crops.items():
            z = cv2.resize(out[i][ya:yb, xa:xb], None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(str(dest / f'{tag}-2x-n{i}.jpg'), z, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        t = cv2.resize(out[i], (640, 360))
        cv2.putText(t, f'n{i}', (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (240, 240, 240), 2)
        thumbs.append(t)
    if thumbs:
        while len(thumbs) % 4:
            thumbs.append(np.zeros((360, 640, 3), np.uint8))
        grid = np.vstack([np.hstack(thumbs[r * 4:(r + 1) * 4]) for r in range(len(thumbs) // 4)])
        cv2.imwrite(str(dest / 'contact-sheet.jpg'), grid, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    # frost before/after at spike frames
    for i in (5, 12, 88, 89, 107, 141):
        if out[i] is None:
            continue
        d = np.clip(np.abs(out[i].astype(np.int16) - src[i].astype(np.int16)) * 3, 0, 255).astype(np.uint8)
        cv2.imwrite(str(dest / f'frost-ba-n{i}.png'),
                    np.hstack([src[i][230:410, 250:560], out[i][230:410, 250:560], d[230:410, 250:560]]))
    tail_strip(out, dest)
    face_compare_grid(out, src, dest)
    mouth_qc(out, src, dest)
    print('preview written', dest, flush=True)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--preview', action='store_true')
    args = ap.parse_args()

    mp4 = P / f'{VER}-test-25s.mp4'
    if not args.preview and mp4.exists():
        raise SystemExit(f'refuse to overwrite {mp4}')

    for label, path, expect in (
        ('v41', Hh.V41_MP4, Hh.V41_SHA),
        ('v41r-ex-mp4', P / 'v41r-ex-test-25s.mp4', V41R_SHA),
        ('v41r-ex-los', SRC, V41R_LOS_SHA),
        ('pcm', PCM, Hh.PCM_SHA),
    ):
        got = sha256(path)
        if got != expect:
            raise SystemExit(f'{label} sha changed: {got}')
    print('frozen shas ok', flush=True)

    print('loading v41r-ex lossless…', flush=True)
    src = Hh.load_span(SRC)
    assert len(src) == NFR
    print('loading cat mattes…', flush=True)
    mattes = Hh.load_span(MATTE, gray=True)

    print('preparing tail / frost / lamp weights…', flush=True)
    tail = prep_tail_padded(SPRITE_FIXED)
    mic = Hh._mic_mask()
    tail_bg, occ = build_tail_bg(src, mattes, tail)
    plate = G.build_plant_plate(src[121], norm_alpha(mattes[121]))
    pdelay = plant_delay_map(plate)
    rest_plant = src[121].astype(np.float32)
    pane = pane_masks()
    cat_union = np.zeros((H, W), np.float32)
    for i in range(0, 241, 3):
        cat_union = np.maximum(cat_union, norm_alpha(mattes[i]))
    cat_union = cv2.GaussianBlur(cat_union, (0, 0), 4.0)
    frost_plate, frost_m = build_frost(src, cat_union)
    w_room, w_bulb = lamp_weights(pane, cat_union)
    ctx = dict(tail=tail, mic=mic, tail_bg=tail_bg, occ=occ,
               plate=plate, pdelay=pdelay, rest_plant=rest_plant,
               frost_plate=frost_plate, frost_m=frost_m, pane=pane,
               w_room=w_room, w_bulb=w_bulb)
    QC.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(QC / 'frost-plate.png'),
                np.clip(frost_plate, 0, 255).astype(np.uint8)[230:410, 250:560])
    vis = src[0].copy()
    vis = vis // 2
    vis[..., 2] = np.clip(vis[..., 2] + (frost_m * 180).astype(np.uint8), 0, 255)
    cv2.imwrite(str(QC / 'frost-mask.png'), vis[230:410, 250:560])
    face_m = frost_m[160:460, 570:950]
    print(f'frost on face crop: mean={face_m.mean():.4f} max={face_m.max():.4f} '
          f'px>0.05={int((face_m > 0.05).sum())}', flush=True)

    if args.preview:
        # one full cycle plus required QC frames
        idx = sorted(set(list(range(0, 241)) + [
            241, 300, 360, 370, 380, 432, 450, 493, 494,
        ]))
        out = process(src, mattes, ctx, preview=idx)
        # fill missing first-cycle frames for plots if we have every 4
        bright, dim = plot_lamp(out, src, QC / 'preview')
        extra = [bright, dim]
        need = [i for i in extra if out[i] is None]
        if need:
            more = process(src, mattes, ctx, preview=need)
            for i in need:
                out[i] = more[i]
        idx2 = sorted(set(idx + [bright, dim]))
        save_preview(out, src, idx2, QC / 'preview', bright, dim)
        save_preview(out, src, idx2, WORK / 'preview', bright, dim)
        for tag, i in (('bright', bright), ('dim', dim)):
            if out[i] is not None:
                cv2.imwrite(str(QC / 'preview' / f'full-{tag}-n{i}.png'), out[i])
        return

    out = process(src, mattes, ctx)
    y0, y1, x0, x1 = TITLE
    bad = [i for i, (a, b) in enumerate(zip(src, out))
           if not np.array_equal(a[y0:y1, x0:x1], b[y0:y1, x0:x1])]
    print('title mismatches' if bad else 'title core byte-identical',
          len(bad) if bad else '', flush=True)

    WORK.mkdir(exist_ok=True)
    los = WORK / f'{VER}-span-890-915-lossless.mkv'
    print('writing lossless…', flush=True)
    Hh.write_lossless(out, los)
    print('encoding mp4…', flush=True)
    Hh.encode_mp4(out, mp4)
    print(VER, 'mp4', sha256(mp4), flush=True)
    print(VER, 'lossless', sha256(los), flush=True)
    shutil.copy2(mp4, WORK / f'{VER}-test-25s.mp4')
    shutil.copy2(Path(__file__), WORK / 'build_v41t_ex.py')
    print('v41 still', sha256(Hh.V41_MP4), flush=True)
    print('v41r-ex still', sha256(P / 'v41r-ex-test-25s.mp4'), flush=True)
    # QC helpers on the processed frames (before they leave memory)
    QC.mkdir(parents=True, exist_ok=True)
    (QC / 'encoded-tmp').mkdir(exist_ok=True)
    plot_lamp(out, src, QC / 'encoded-tmp')
    tail_strip(out, QC / 'encoded-tmp')
    face_compare_grid(out, src, QC / 'encoded-tmp')
    mouth_qc(out, src, QC / 'encoded-tmp')


if __name__ == '__main__':
    main()
