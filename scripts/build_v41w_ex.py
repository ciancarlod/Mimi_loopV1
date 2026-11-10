#!/usr/bin/env python3
"""v41w-ex: fix v41v-ex tail cut-out / squared tip / haunch smear / leftover hook, and meow cheek ripples.

Post-pass on frozen v41u-ex lossless. Jaw is a rigid hinge rotation about
the TMJ (no side shift). Eyes and sockets stay fixed above the cheekbone.
Ears are copied from u-ex. Tail is a 20-segment caudal chain, stiff at the
base and loose at the tip, drawn as a solid tapered tube in front of the plant.

Does not edit earlier builders or encodes. Audio is the approved v41 PCM
(mux only). Title is copied through. Snow/steam/frost/lamp inherited from
u-ex and not re-touched.
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
import build_v41t_ex as T
import build_v41u_ex as U

VER = 'v41w-ex'
WORK = P / VER
QC = P / 'qc' / VER
SRC_U = P / 'v41u-ex' / 'v41u-ex-span-890-915-lossless.mkv'
SRC_Q = P / 'v41q-ex' / 'v41q-ex-span-890-915-lossless.mkv'
MATTE = Hh.MATTE
PCM = Hh.PCM

V41_SHA = Hh.V41_SHA
V41Q_SHA = 'be65528916d68b99cc64546d865388f824656b3975adb8f861f10d03e548afcd'
V41U_SHA = 'e14aa352424f4cc729c254f255b04907cc770063216083e8e970be1858e92560'
V41U_LOS_SHA = '018e7829b02b3eb5343a3b5bd243fae2077306bbd9241bfd1d465995276e5dc5'
V41V_SHA = '22bc91be4db9db85e15919258cf57724901f7b209c618dbe480b0c755fdf0bf6'

NFR, W, H, NCYCLE = 600, 1280, 720, 241
TITLE = Hh.TITLE
Yg, Xg = np.mgrid[0:H, 0:W].astype(np.float32)

# Quick flicks ~53 px tip, slow ~72 px (length-preserving chain, rest still).
BASE_DEG = (9.0, 10.2)
MID_DEG = (12.5, 14.5)
TIP_DEG = (22.0, 18.5)
MID_LAG, TIP_LAG = 2.2, 5.0

# Dusk fur (BGR), sampled from the u-ex haunch/chest and biased slightly
# brighter so the tail reads as a solid cream/ginger tube in the shadows.
CREAM = np.array([62.0, 104.0, 126.0], np.float32)
GINGER = np.array([42.0, 86.0, 150.0], np.float32)
TABBY = np.array([28.0, 48.0, 62.0], np.float32)
RIM = np.array([48.0, 78.0, 96.0], np.float32)

# Mouth / jaw — TMJ hinge (see anatomy notes in OVERNIGHT-LOG)
# Just in front of and below the ear base, level with the eye corners,
# behind the cheek. Sagittal rotation: x is unchanged (no side shift).
Y_TMJ = 292.0
Y_CHEEK = 332.0         # zero warp at/above the cheekbone (eyes + sockets)
Y_UPPER = 392.0         # upper lip / nose stay on the skull
Y_MAND = 408.0          # full mandible rotation below the closed mouth line
MEOW_A = (130, 138, 145)  # start, peak, end (medium, ~15°)
MEOW_B = (425, 432, 440)  # start, peak, end (big, ~25°)
DEG_A, DEG_B = 15.0, 25.0
# assumed rest z / y of a mandible point (chin is rostral to the TMJ)
Z_OVER_Y = 0.80
WIDE_A, WIDE_B = 36.0, 46.0


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for ch in iter(lambda: f.read(1 << 20), b''):
            h.update(ch)
    return h.hexdigest()


def luma(im):
    im = im.astype(np.float32)
    return 0.114 * im[..., 0] + 0.587 * im[..., 1] + 0.299 * im[..., 2]


# ---------------- tail ----------------
def deform_flicks(rest, ph):
    dxy = np.diff(rest, axis=0)
    seglen = np.hypot(dxy[:, 0], dxy[:, 1])
    rest_ang = np.arctan2(dxy[:, 1], dxy[:, 0])
    arc = np.concatenate([[0.0], np.cumsum(seglen)])
    s = arc / (arc[-1] + 1e-8)
    pts = np.zeros_like(rest)
    pts[0] = rest[0]
    for i in range(len(seglen)):
        sm = 0.5 * (s[i] + s[i + 1])
        # Relative joint rotation: base vertebrae are stiffer (small dθ),
        # distal joints looser. A flick is a wave from base to tip with lag.
        pb = T._flick_deg(ph, 0.0, BASE_DEG[0], BASE_DEG[1]) * (0.15 + 0.85 * sm ** 1.35)
        pm = T._flick_deg(ph, MID_LAG, MID_DEG[0], MID_DEG[1]) * T.sstep((sm - 0.18) / 0.45)
        pt = T._flick_deg(ph, TIP_LAG, TIP_DEG[0], TIP_DEG[1]) * (sm ** 1.75)
        ang = rest_ang[i] + np.radians(pb + pm + pt)
        pts[i + 1] = pts[i] + seglen[i] * np.array([np.cos(ang), np.sin(ang)], np.float32)
    return pts


def width_profile(s):
    """Consistent tapered tube. Depends only on arc length, not on pose."""
    s = np.clip(np.asarray(s, np.float32), 0, 1)
    root, mid, tip = 40.0, 34.0, 20.0
    # root -> mid by s=0.45, mid -> tip, round the tip cap
    w = np.where(
        s < 0.45,
        root + (mid - root) * T.sstep(s / 0.45),
        mid + (tip - mid) * T.sstep((s - 0.45) / 0.55),
    )
    return w


def shade_tail(s, sd, w):
    """Opaque cream/ginger fur with stable tabby ticks and a darker rim."""
    s = np.clip(s, 0, 1)
    # ginger band along the outer/top half of the mid-tail (sprite-like)
    ging = T.sstep((s - 0.12) / 0.10) * (1.0 - T.sstep((s - 0.62) / 0.16))
    # more ginger on the "back" side of the tube (sd > 0 is one side)
    side = T.sstep((sd) / (0.35 * w + 1.0))
    ging = np.clip(ging * (0.55 + 0.55 * side), 0, 1)
    col = CREAM[None, None, :] * (1.0 - ging[..., None]) + GINGER[None, None, :] * ging[..., None]
    # tabby ticks: narrow dark stripes at fixed s, stronger on the top edge
    ticks = np.zeros_like(s)
    for c, sig in ((0.22, 0.018), (0.30, 0.016), (0.38, 0.016), (0.46, 0.015),
                   (0.54, 0.014), (0.62, 0.014)):
        ticks = ticks + np.exp(-0.5 * ((s - c) / sig) ** 2)
    ticks = np.clip(ticks, 0, 1) * T.sstep((-sd) / (0.40 * w + 1.0)) * 0.70
    col = col * (1.0 - ticks[..., None]) + TABBY[None, None, :] * ticks[..., None]
    # slightly lighter highlight along the upper third
    hi = T.sstep((-sd - 0.15 * w) / (0.35 * w + 1.0)) * 0.10
    col = col * (1.0 + hi[..., None])
    # darker rim in the last ~3.5 px
    rim = T.sstep((np.abs(sd) - (w - 3.8)) / 2.4)
    col = col * (1.0 - 0.72 * rim[..., None]) + RIM[None, None, :] * (0.72 * rim[..., None])
    return np.clip(col, 0, 255)


def plant_only_mask(img):
    """Static-ish green leaf test; used so the plant reaction never warps the tail."""
    im = img.astype(np.float32)
    B, Gc, R = im[..., 0], im[..., 1], im[..., 2]
    Y = luma(im)
    leaf = (Gc > R + 6) & (Gc > B + 2) & (Y < 110) & (Xg > 1000) & (Yg > 470)
    m = leaf.astype(np.float32)
    m = cv2.GaussianBlur(m, (0, 0), 1.2)
    return np.clip(m, 0, 1)


def apply_plant_reaction(img, plate, ph, tip):
    """Small damped leaf bend only where the tail brushes; plant pixels only."""
    live = np.clip(plate, 0, 1)
    if live.max() < 0.05:
        return img
    # drive from the flicks, delayed, and only near the current tip
    sig = (0.90 * T.flick_env(ph - T.SLOW_F[0] - 3.0, T.SLOW_F[1], T.SLOW_F[2], 0.08)
           + 0.18 * T.flick_env(ph - T.QUICK_A[0] - 2.0, T.QUICK_A[1], T.QUICK_A[2], 0.06)
           + 0.18 * T.flick_env(ph - T.QUICK_B[0] - 2.0, T.QUICK_B[1], T.QUICK_B[2], 0.06))
    near = np.exp(-0.5 * ((Xg - float(tip[0])) ** 2 + (Yg - float(tip[1])) ** 2) / (95.0 ** 2))
    A = (sig * live * near).astype(np.float32)
    if float(np.max(np.abs(A))) < 0.03:
        return img
    ang = np.radians(2.4) * A
    trans = 3.5 * A
    PUSH = np.array([0.92, 0.39], np.float32)
    PUSH = PUSH / np.hypot(*PUSH)
    stem = G.STEM
    vx, vy = Xg - stem[0], Yg - stem[1]
    ca, sa = np.cos(ang), np.sin(ang)
    sx = stem[0] + ca * vx + sa * vy - (trans * PUSH[0])
    sy = stem[1] - sa * vx + ca * vy - (trans * PUSH[1])
    y0, y1, x0, x1 = 450, H, 1000, W
    mapx, mapy = Xg.copy(), Yg.copy()
    mapx[y0:y1, x0:x1] = sx[y0:y1, x0:x1]
    mapy[y0:y1, x0:x1] = sy[y0:y1, x0:x1]
    imgf = img.astype(np.float32)
    warped = cv2.remap(imgf, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    # only move plant pixels; never the body or the (not-yet-drawn) tail region we'll overwrite
    wa = cv2.remap(live, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    src_leaf = plant_only_mask(img)
    dst_leaf = plant_only_mask(np.clip(warped, 0, 255))
    wa = np.clip(wa * np.maximum(src_leaf, dst_leaf), 0, 1)
    out = imgf * (1.0 - wa[..., None]) + warped * wa[..., None]
    return np.clip(out, 0, 255)



def inherited_hook_occ(src_u, mattes):
    """Pale leftover hook in the leaf corner, unioned over several u-ex frames."""
    acc = np.zeros((H, W), np.float32)
    for i in (0, 12, 40, 80, 116, 120, 160, 180, 200, 230):
        im = src_u[i].astype(np.float32)
        a = T.norm_alpha(mattes[i])
        B, Gc, R = im[..., 0], im[..., 1], im[..., 2]
        Y = luma(im)
        # cream / pale fur, not a green leaf, in the plant corner
        cream = ((Y > 55) & (R > Gc - 8) & (R > B + 8) & (Gc < R + 18)
                 & (Xg > 990) & (Yg > 520) & (Yg < 720))
        # also any mid-luma beige blob the sprite does not own
        acc = np.maximum(acc, cream.astype(np.float32) * (1.0 - T.sstep((a - 0.55) / 0.20)))
    acc = cv2.dilate(acc, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    acc = cv2.GaussianBlur(acc, (0, 0), 2.0)
    return np.clip(acc, 0, 1)


def build_static_leaf_plate(src_q, mattes, seed_bg):
    """Static plant fill for leftover hook pixels only, held fixed.

    Start from T.build_tail_bg. In the lower-right leaves (not the chair or
    haunch), replace remaining cream/pale-fur with a median of leafy q-ex
    samples, then one Telea pass on any leftover holes.
    """
    bg = seed_bg.astype(np.float32).copy()
    acc = np.zeros((H, W, 3), np.float64)
    wsum = np.zeros((H, W), np.float64)
    # only the plant foliage, right of the hook
    roi = (Xg > 1088) & (Yg > 560)
    for i in range(241):
        im = src_q[i].astype(np.float32)
        a = T.norm_alpha(mattes[i])
        B, Gc, R = im[..., 0], im[..., 1], im[..., 2]
        Y = luma(im)
        leaf = roi & (Gc > R + 4) & (Gc > B) & (Y < 125) & (a < 0.20)
        if not np.any(leaf):
            continue
        acc[leaf] += im[leaf]
        wsum[leaf] += 1.0
    have = wsum > 6.0
    bg[have] = (acc[have] / wsum[have, None]).astype(np.float32)
    B, Gc, R = bg[..., 0], bg[..., 1], bg[..., 2]
    Y = luma(bg)
    # leftover pale hook only, inside the leaf roi, not brown chair
    cream = roi & (Y > 58) & (R > Gc - 2) & (R > B + 8) & (Gc < R + 20)
    hole = cream.astype(np.uint8)
    hole = cv2.dilate(hole, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    hole = hole * roi.astype(np.uint8)
    if int(hole.sum()) > 20:
        filled = cv2.inpaint(np.clip(bg, 0, 255).astype(np.uint8),
                             hole * 255, 7, cv2.INPAINT_TELEA).astype(np.float32)
        m = cv2.GaussianBlur(hole.astype(np.float32), (0, 0), 1.2)
        m = np.clip(m, 0, 1) * roi.astype(np.float32)
        bg = bg * (1.0 - m[..., None]) + filled * m[..., None]
    return np.clip(bg, 0, 255)


def apply_solid_tail(img, cat_a, ph, tail, bg, occ, plate, mic):
    """Clear the inherited tangled hook, restore plant, then draw an opaque
    painted tail (vertebral chain) fully in front of the leaves, with a rim.
    """
    imgf = img.astype(np.float32)
    rest = tail['rest']
    gate = T.sstep((Yg - 500.0) / 16.0) * T.sstep((Xg - 900.0) / 16.0)
    occ_s = np.clip(occ * gate, 0, 1)
    cleared = imgf * (1.0 - occ_s[..., None]) + bg * occ_s[..., None]
    live = np.clip(plate, 0, 1) * gate
    live = live * (1.0 - T.sstep((np.clip(cat_a, 0, 1) - 0.25) / 0.25))
    live = cv2.GaussianBlur(live, (0, 0), 1.0)
    rest_mix = live * (1.0 - np.clip(cat_a, 0, 1))
    cleared = cleared * (1.0 - rest_mix[..., None]) + bg * rest_mix[..., None]

    defp = deform_flicks(rest, ph)
    tip = defp[-1]
    cleared = apply_plant_reaction(cleared, live, ph, tip)

    # Inverse-map the rest sprite along the bent 20-vertebra chain.
    # At rest, paste the sprite as-is (identity remap is unsafe past the
    # endpoints: sd~0 along the last tangent would paint an infinite hose).
    pad = 90
    xs = np.concatenate([rest[:, 0], defp[:, 0]])
    ys = np.concatenate([rest[:, 1], defp[:, 1]])
    x0, x1 = int(max(0, xs.min() - pad)), int(min(W, xs.max() + pad + 1))
    y0, y1 = int(max(0, ys.min() - pad)), int(min(H, ys.max() + pad + 1))
    moved = float(np.max(np.abs(defp - rest)))
    if moved < 0.4:
        rgb = tail['world_rgb'][y0:y1, x0:x1].copy()
        al = tail['world_a'][y0:y1, x0:x1].copy()
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        arc_d, tot_d = Hh._arc(defp)
        s, sd, tx, ty = Hh._closest(xx, yy, defp, arc_d, tot_d)
        w = np.interp(np.clip(s, 0, 1),
                       np.linspace(0, 1, len(tail['widths'])),
                       tail['widths'])
        # still cap ends and drop baked leaves
    else:
        arc_r, tot_r = Hh._arc(rest)
        arc_d, tot_d = Hh._arc(defp)
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        s, sd, tx, ty = Hh._closest(xx, yy, defp, arc_d, tot_d)
        tg = np.zeros_like(rest)
        tg[1:-1] = rest[2:] - rest[:-2]
        tg[0] = rest[1] - rest[0]
        tg[-1] = rest[-1] - rest[-2]
        ln = np.sqrt((tg ** 2).sum(1)) + 1e-8
        tg = tg / ln[:, None]
        nrm = np.stack([-tg[:, 1], tg[:, 0]], 1)
        rx = np.interp(s * tot_r, arc_r, rest[:, 0])
        ry = np.interp(s * tot_r, arc_r, rest[:, 1])
        nx = np.interp(s * tot_r, arc_r, nrm[:, 0])
        ny = np.interp(s * tot_r, arc_r, nrm[:, 1])
        mapx = (rx + sd * nx).astype(np.float32)
        mapy = (ry + sd * ny).astype(np.float32)
        rgb = cv2.remap(tail['world_rgb'], mapx, mapy, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT)
        al = cv2.remap(tail['world_a'], mapx, mapy, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT)
        w = np.interp(np.clip(s, 0, 1),
                       np.linspace(0, 1, len(tail['widths'])),
                       tail['widths'])
    # Rounded tip cap: past the tip vertex, keep a circular end (hypot), not
    # a box in (along, sd) which squared the nub off at n120/n180. Root is
    # not capped this way — it tucks behind the haunch.
    cx = np.interp(s * tot_d, arc_d, defp[:, 0])
    cy = np.interp(s * tot_d, arc_d, defp[:, 1])
    along = (xx - cx) * tx + (yy - cy) * ty
    al = np.clip(al, 0, 1)
    # width clip with a soft fur edge, not a hard cookie-cutter
    al = al * T.sstep((w + 10.0 - np.abs(sd)) / 4.0)
    tipz = T.sstep((s - 0.80) / 0.10)
    rcap = np.hypot(np.maximum(along, 0.0), sd)
    al = al * (1.0 - tipz + tipz * T.sstep(((w + 8.0) - rcap) / 3.0))
    # drop only clearly green leaf pixels baked into the sprite, not olive fur
    leaf = (rgb[..., 1] > rgb[..., 2] + 14) & (rgb[..., 1] > rgb[..., 0] + 8) & (rgb[..., 1] > 40)
    al = al * (1.0 - 0.95 * leaf.astype(np.float32))
    # lift mid-alpha just enough to stay opaque over leaves, keep 2-3 px AA
    al = np.clip((al - 0.04) / 0.82, 0, 1)
    al = T.sstep(al)
    # subtle shading, not a drawn outline
    edge = al * (1.0 - T.sstep((w - 2.4 - np.abs(sd)) / 3.2)) * 0.16
    rgb = rgb * (1.0 - edge[..., None]) + RIM[None, None, :] * edge[..., None]
    la = np.zeros((H, W), np.float32)
    layer = np.zeros((H, W, 3), np.float32)
    la[y0:y1, x0:x1] = al
    layer[y0:y1, x0:x1] = rgb
    # tail fully in front of the leaves it touches
    comp = cleared * (1.0 - la[..., None]) + layer * la[..., None]
    # Tuck the root behind the haunch with a sharp body edge (1–2 px AA),
    # no wide Gaussian smear along the haunch.
    cat_edge = np.clip(cat_a, 0, 1)
    haunch = (T.sstep((cat_edge - 0.38) / 0.16)
              * T.sstep((1022.0 - Xg) / 10.0)
              * T.sstep((Yg - 515.0) / 16.0))
    haunch = np.clip(haunch, 0, 1)
    comp = comp * (1.0 - haunch[..., None]) + imgf * haunch[..., None]
    comp = imgf * mic[..., None] + comp * (1.0 - mic[..., None])
    area = float((la > 0.5).sum())
    return np.clip(comp, 0, 255), tip, area, la


# ---------------- mouth / jaw / eyes ----------------
def meow_amount(i):
    """0..1 envelope, 4-6 frame ease in/out, bigger window for the big meow."""
    def env(start, peak, end):
        if i <= start or i >= end:
            return 0.0
        if i <= peak:
            u = (i - start) / max(peak - start, 1)
        else:
            u = (end - i) / max(end - peak, 1)
        # smoothstep
        return float(u * u * (3.0 - 2.0 * u))
    a = env(*MEOW_A)
    b = env(*MEOW_B)
    return a, b  # medium, big (at most one is nonzero)


def eye_mask():
    """Stable ellipses covering both eyes; never warped or painted."""
    m = np.zeros((H, W), np.float32)
    # left (viewer's) eye, covers both the squint pose and the wide n432 pose
    cv2.ellipse(m, (650, 272), (58, 42), -8, 0, 360, 1, -1)
    cv2.ellipse(m, (800, 278), (62, 44), 8, 0, 360, 1, -1)
    m = cv2.GaussianBlur(m, (0, 0), 3.0)
    # hard cut above the nose line is implicit: these sit well above the cheekbone
    return np.clip(m, 0, 1)


def find_nose(img):
    """Tight pink nose in the muzzle; fallback to a stable default."""
    y0, y1, x0, x1 = 340, 390, 690, 750
    sub = img[y0:y1, x0:x1].astype(np.float32)
    B, G, R = sub[..., 0], sub[..., 1], sub[..., 2]
    Y = luma(sub)
    nose = ((R > G + 18) & (R > B + 22) & (R > 85) & (Y > 60) & (Y < 150)
            & ((R - G) > 14)).astype(np.uint8)
    nose = cv2.morphologyEx(nose, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    if int(nose.sum()) < 12:
        return np.array([718.0, 368.0], np.float32)
    ys, xs = np.where(nose > 0)
    return np.array([float(np.median(xs) + x0), float(np.median(ys) + y0)], np.float32)


def ear_mask():
    """u-ex ears stay put — lock both pinnae, including the bases."""
    m = np.zeros((H, W), np.float32)
    cv2.ellipse(m, (575, 200), (70, 85), -18, 0, 360, 1, -1)
    cv2.ellipse(m, (900, 205), (68, 82), 16, 0, 360, 1, -1)
    return np.clip(cv2.GaussianBlur(m, (0, 0), 3.0), 0, 1)



def close_old_mouth(img, eyes, ears):
    """Fill the inherited slit so the new opening is not a double-exposure."""
    imgf = img.astype(np.float32)
    y0, y1, x0, x1 = 378, 460, 668, 780
    sub = imgf[y0:y1, x0:x1]
    Y = luma(sub)
    B, G, R = sub[..., 0], sub[..., 1], sub[..., 2]
    # tight muzzle only: dark cavity or saturated darker-than-fur pink tongue
    # (ginger cheek is also R>G, so require a large red surplus and lower Y)
    pink = (R > G + 18) & (R > B + 10) & (Y < 140) & (Y > 30)
    dark = (Y < 105) & (Y > 12) & (R + 12 > G)
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    ell = ((((xx - 718.0) / 38.0) ** 2) + (((yy - 422.0) / 22.0) ** 2) < 1.0)
    op = ((pink | dark | ell) & (yy > 388) & (yy < 450) & (xx > 680) & (xx < 765)).astype(np.uint8)
    op = cv2.dilate(op, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    if int(op.sum()) < 8:
        return img
    fill = cv2.inpaint(np.clip(sub, 0, 255).astype(np.uint8),
                       op * 255, 6, cv2.INPAINT_TELEA).astype(np.float32)
    m = cv2.GaussianBlur(op.astype(np.float32), (0, 0), 1.2)
    m = np.clip(m, 0, 1)
    sub2 = sub * (1.0 - m[..., None]) + fill * m[..., None]
    out = imgf.copy()
    out[y0:y1, x0:x1] = sub2
    lock = np.clip(np.maximum(eyes, ears), 0, 1)
    out = out * (1.0 - lock[..., None]) + imgf * lock[..., None]
    return np.clip(out, 0, 255)


def chin_mask(nose):
    """Soft oval on the chin and upper throat only. Zero on cheeks and pads."""
    cx = float(nose[0])
    # centre well below the lower lip; taper like a chin, not a rectangle
    cy, ax, ay = 505.0, 28.0, 48.0
    r2 = ((Xg - cx) / ax) ** 2 + ((Yg - cy) / ay) ** 2
    m = T.sstep((1.0 - r2) / 0.22)
    m = m * T.sstep((Yg - 452.0) / 10.0)          # start below the lower lip
    m = m * (1.0 - T.sstep((Yg - 555.0) / 10.0))
    return np.clip(m, 0, 1)


def apply_chin_drop(img, dy, nose, eyes, ears):
    """Small rigid downward translation of the chin only. Cheeks stay put.

    The u-ex mouth (tongue, fangs, upper lip, philtrum) is locked. No sliver
    bar, no rectangular mask. Feather only into the throat fur below.
    """
    if dy < 0.4:
        return img
    imgf = img.astype(np.float32)
    m = chin_mask(nose) * (1.0 - eyes) * (1.0 - ears) * 0.70
    mapx = Xg
    mapy = (Yg - float(dy) * m).astype(np.float32)
    mapy = np.clip(mapy, 0, H - 1)
    moved = cv2.remap(imgf, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    out = imgf * (1.0 - m[..., None]) + moved * m[..., None]
    # locks: eyes, ears, everything above the lower lip, the cheeks / pads
    lock = np.clip(np.maximum(eyes, ears), 0, 1)
    lock = np.maximum(lock, 1.0 - T.sstep((Yg - 438.0) / 5.0))
    lock = np.maximum(lock, T.sstep((float(nose[0]) - 52.0 - Xg) / 4.0))
    lock = np.maximum(lock, T.sstep((Xg - (float(nose[0]) + 52.0)) / 4.0))
    out = out * (1.0 - lock[..., None]) + imgf * lock[..., None]
    return np.clip(out, 0, 255)


def paint_meow(img, amt, wide, tall, nose, eyes, ears):
    """Fill the hinge-opened gap: dark-red interior, pink tongue, white fangs, lip line."""
    if amt < 0.08:
        return img
    imgf = img.astype(np.float32)
    cx, cy = float(nose[0]), float(nose[1] + 22.0 + 0.20 * tall)
    # opening ellipse in a local map (slightly pointed at the corners via a squircle mix)
    xx = (Xg - cx) / (wide * 0.50 + 1e-6)
    yy = (Yg - cy) / (tall * 0.50 + 1e-6)
    r2 = xx ** 2 + yy ** 2
    # slightly flatter bottom (jaw)
    yy2 = np.where(yy > 0, yy * 0.92, yy * 1.05)
    r2 = xx ** 2 + yy2 ** 2
    inside = T.sstep((1.0 - r2) / 0.18)
    # tight to the muzzle: kill anything far from the opening or above the nose
    inside = inside * T.sstep((Yg - (Y_UPPER + 4.0)) / 8.0)
    inside = inside * (1.0 - eyes)
    # hard muzzle box — never the cheek or chin sides
    inside = inside * T.sstep((cx + wide * 0.52 - Xg) / 3.0)
    inside = inside * T.sstep((Xg - (cx - wide * 0.52)) / 3.0)
    inside = inside * T.sstep((cy + tall * 0.62 - Yg) / 3.0)
    if float(inside.max()) < 0.05:
        return img
    # colours (BGR, dusk-warm, not mauve)
    interior = np.array([28.0, 32.0, 72.0], np.float32)
    tongue = np.array([56.0, 78.0, 168.0], np.float32)
    lip = np.array([18.0, 22.0, 36.0], np.float32)
    fang = np.array([210.0, 220.0, 228.0], np.float32)
    # tongue: lower half oval
    txx = (Xg - cx) / (wide * 0.22 + 1e-6)
    tyy = (Yg - (cy + tall * 0.16)) / (tall * 0.22 + 1e-6)
    tongue_m = T.sstep((1.0 - (txx ** 2 + tyy ** 2)) / 0.22) * inside
    # fangs: two small triangles on the upper lip
    fangs = np.zeros((H, W), np.float32)
    for fx in (-0.22, 0.22):
        px = cx + fx * wide * 0.70
        py = cy - tall * 0.06
        fxn = (Xg - px) / 4.2
        fyn = (Yg - py) / 6.5
        fangs = np.maximum(fangs, T.sstep((1.0 - (np.abs(fxn) + fyn)) / 0.35) * (fyn > 0).astype(np.float32))
    fangs = fangs * inside
    # lip ring: band just inside the opening edge
    lip_m = inside * (1.0 - T.sstep((1.0 - r2) / 0.10)) * T.sstep((1.0 - r2) / 0.28)
    col = np.zeros((H, W, 3), np.float32)
    col += interior
    col = col * (1.0 - 0.90 * tongue_m[..., None]) + tongue * (0.90 * tongue_m[..., None])
    col = col * (1.0 - 0.92 * fangs[..., None]) + fang * (0.92 * fangs[..., None])
    col = col * (1.0 - 0.85 * lip_m[..., None]) + lip * (0.85 * lip_m[..., None])
    a = np.clip(inside, 0, 1)
    a = a * (1.0 - ears) * T.sstep((Yg - Y_CHEEK) / 6.0)
    out = imgf * (1.0 - a[..., None]) + col * a[..., None]
    lock = np.clip(np.maximum(eyes, ears), 0, 1)
    out = out * (1.0 - lock[..., None]) + imgf * lock[..., None]
    return np.clip(out, 0, 255)


def apply_meow(img, i, eyes, ears):
    a, b = meow_amount(i)
    amt = max(a, b)
    if amt < 0.02:
        return img
    # v41v-ex's wide TMJ remap rippled the cheek and whisker-pad fur and
    # tore the lower lip. Fall back: keep the u-ex mouth, drop only the
    # chin as a rigid translation, and hold a clean lower-lip edge.
    dy = 4.0 * a + 6.5 * b
    nose = np.array([718.0, 368.0], np.float32)  # stable; do not chase a stretched nose
    out = apply_chin_drop(img, dy, nose, eyes, ears)
    return np.clip(out, 0, 255)


def process_one(src_u, cat_a, ph, ctx, i):
    img = src_u
    img, tip, area, la = apply_solid_tail(
        img, cat_a, ph, ctx['tail'], ctx['tail_bg'], ctx['occ'],
        ctx['plate'], ctx['mic'])
    img = apply_meow(img, i, ctx['eyes'], ctx['ears'])
    img = np.clip(img, 0, 255).astype(np.uint8)
    y0, y1, x0, x1 = TITLE
    img = img.copy()
    img[y0:y1, x0:x1] = src_u[y0:y1, x0:x1]
    return img, tip, area


def process(src_u, mattes, ctx, preview=None):
    want = set(preview) if preview else set(range(NFR))
    out = [None] * NFR
    areas = np.full(NFR, np.nan, np.float32)
    tips = [None] * NFR
    for i in range(NFR):
        if i not in want:
            continue
        ph = Hh.phase_of(i)
        out[i], tips[i], areas[i] = process_one(src_u[i], T.norm_alpha(mattes[i]), ph, ctx, i)
        if preview or i % 25 == 0:
            print(f'  n{i} ph={ph} tip={tips[i]} area={areas[i]:.0f}', flush=True)
    return out, areas, tips


# ---------------- QC ----------------
def tail_strip_2x(out, dest, every=6):
    dest.mkdir(parents=True, exist_ok=True)
    cells = []
    for i in range(0, 241, every):
        if out[i] is None:
            continue
        c = out[i][380:720, 880:1280]
        z = cv2.resize(c, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        cv2.putText(z, f'n{i}', (8, 28), cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255), 2)
        cells.append(z)
    if not cells:
        return
    cols = 5
    while len(cells) % cols:
        cells.append(np.zeros_like(cells[0]))
    rows = [np.hstack(cells[k:k + cols]) for k in range(0, len(cells), cols)]
    path = dest / 'tail-strip-2x.png'
    cv2.imwrite(str(path), np.vstack(rows))
    print('tail strip', path)


def meow_seq(out, dest):
    dest.mkdir(parents=True, exist_ok=True)
    for tag, rng in (('138', range(130, 146, 2)), ('432', range(425, 441, 2))):
        cells = []
        for i in rng:
            if out[i] is None:
                continue
            c = cv2.resize(out[i][330:500, 620:860], None, fx=3, fy=3,
                           interpolation=cv2.INTER_NEAREST)
            cv2.putText(c, f'n{i}', (8, 28), cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255), 2)
            cells.append(c)
        if cells:
            path = dest / f'meow-seq-{tag}.png'
            cv2.imwrite(str(path), np.hstack(cells))
            print('meow seq', path)


def eye_seq(out, dest):
    dest.mkdir(parents=True, exist_ok=True)
    boxes = {
        ('138', 'L'): (235, 310, 600, 700),
        ('138', 'R'): (250, 340, 740, 860),
        ('432', 'L'): (210, 300, 600, 710),
        ('432', 'R'): (210, 305, 750, 870),
    }
    ranges = {'138': range(126, 147), '432': range(422, 443)}
    for (tag, eye), box in boxes.items():
        y0, y1, x0, x1 = box
        cells = []
        for i in ranges[tag]:
            if out[i] is None:
                continue
            c = cv2.resize(out[i][y0:y1, x0:x1], None, fx=3, fy=3,
                           interpolation=cv2.INTER_NEAREST)
            cv2.putText(c, f'n{i}', (4, 18), cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 255, 255), 1)
            cells.append(c)
        if not cells:
            continue
        cols = 7
        while len(cells) % cols:
            cells.append(np.zeros_like(cells[0]))
        rows = [np.hstack(cells[k:k + cols]) for k in range(0, len(cells), cols)]
        path = dest / f'eyes-{eye}-{tag}.png'
        cv2.imwrite(str(path), np.vstack(rows))
        print('eyes', path)


def before_after(out, src_u, dest):
    dest.mkdir(parents=True, exist_ok=True)
    # tail n0 / n120 / n180
    tails = []
    for i in (0, 120, 180):
        if out[i] is None:
            continue
        a = src_u[i][380:720, 880:1280]
        b = out[i][380:720, 880:1280]
        pair = np.hstack([a, b])
        cv2.putText(pair, f'u n{i}', (8, 24), cv2.FONT_HERSHEY_PLAIN, 1.6, (255, 255, 255), 2)
        cv2.putText(pair, f'{VER} n{i}', (a.shape[1] + 8, 24), cv2.FONT_HERSHEY_PLAIN, 1.6, (255, 255, 255), 2)
        tails.append(pair)
    if tails:
        path = dest / 'tail-u-vs-new.png'
        cv2.imwrite(str(path), np.vstack(tails))
        print('tail ba', path)
    # mouth n432 and n138
    mouths = []
    for i in (138, 432):
        if out[i] is None:
            continue
        a = cv2.resize(src_u[i][330:500, 620:860], None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
        b = cv2.resize(out[i][330:500, 620:860], None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
        pair = np.hstack([a, b])
        cv2.putText(pair, f'u n{i}', (8, 28), cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255), 2)
        cv2.putText(pair, f'{VER} n{i}', (a.shape[1] + 8, 28), cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255), 2)
        mouths.append(pair)
    if mouths:
        path = dest / 'mouth-u-vs-new-n138-n432.png'
        cv2.imwrite(str(path), np.vstack(mouths))
        print('mouth ba', path)


def face_and_chest(out, src_u, dest):
    dest.mkdir(parents=True, exist_ok=True)
    # reuse t-ex grids vs u-ex for face (should stay close except mouth)
    try:
        T.face_compare_grid(out, src_u, dest)
    except Exception as e:
        print('face grid skip', e)
    # chest vs u-ex (look for rectangles)
    y0, y1, x0, x1 = 300, 520, 680, 900
    tiles = []
    for i in range(0, 241, 10):
        if out[i] is None:
            continue
        c = cv2.resize(out[i][y0:y1, x0:x1], (200, 200), interpolation=cv2.INTER_AREA)
        cv2.putText(c, f'n{i}', (6, 18), cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 255, 255), 1)
        tiles.append(c)
    if tiles:
        while len(tiles) % 5:
            tiles.append(np.zeros_like(tiles[0]))
        rows = [np.hstack(tiles[k:k + 5]) for k in range(0, len(tiles), 5)]
        path = dest / 'chest-grid.png'
        cv2.imwrite(str(path), np.vstack(rows))
        print('chest', path)


def area_report(areas, dest, tips=None, rest_tip=None):
    dest.mkdir(parents=True, exist_ok=True)
    v = areas[:241]
    ok = np.isfinite(v)
    if not ok.any():
        return
    mean = float(np.nanmean(v))
    ptp = float(np.nanmax(v) - np.nanmin(v))
    print(f'tail silhouette area (n0-240): mean={mean:.0f} ptp={ptp:.0f} '
          f'({100 * ptp / max(mean, 1):.1f}%) min={np.nanmin(v):.0f} max={np.nanmax(v):.0f}')
    if tips is not None and rest_tip is not None:
        d = []
        for i in range(241):
            if tips[i] is None:
                d.append(np.nan)
            else:
                d.append(float(np.linalg.norm(tips[i] - rest_tip)))
        d = np.array(d, np.float32)
        def pk(a, b):
            sl = d[a:b]
            if not np.isfinite(sl).any():
                return None
            return float(np.nanmax(sl)), a + int(np.nanargmax(np.where(np.isfinite(sl), sl, -1)))
        print('tip travel Q1', pk(18, 46), 'SLOW', pk(75, 120), 'Q2', pk(155, 183),
              'n0', d[0] if np.isfinite(d[0]) else None, 'n240', d[240] if np.isfinite(d[240]) else None)
    np.save(dest / 'tail-area.npy', v)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--preview', action='store_true')
    args = ap.parse_args()

    mp4 = P / f'{VER}-test-25s.mp4'
    if not args.preview and mp4.exists():
        raise SystemExit(f'refuse to overwrite {mp4}')

    for label, path, expect in (
        ('v41', Hh.V41_MP4, V41_SHA),
        ('v41q-ex-mp4', P / 'v41q-ex-test-25s.mp4', V41Q_SHA),
        ('v41u-ex-mp4', P / 'v41u-ex-test-25s.mp4', V41U_SHA),
        ('v41u-ex-los', SRC_U, V41U_LOS_SHA),
        ('v41v-ex-mp4', P / 'v41v-ex-test-25s.mp4', V41V_SHA),
        ('pcm', PCM, Hh.PCM_SHA),
    ):
        got = sha256(path)
        if got != expect:
            raise SystemExit(f'{label} sha changed: {got}')
    print('frozen shas ok', flush=True)

    print('loading u-ex / q-ex / mattes…', flush=True)
    src_u = Hh.load_span(SRC_U)
    src_q = Hh.load_span(SRC_Q)
    mattes = Hh.load_span(MATTE, gray=True)
    assert len(src_u) == NFR

    print('preparing tail plates…', flush=True)
    tail = T.prep_tail_padded(T.SPRITE_FIXED)
    # 20 caudal vertebrae (cats have 18–23), steadily tapering
    rest = T.resample_poly(tail['rest'], 20)
    mic = Hh._mic_mask()
    # background from q-ex (no t-ex tangled hook) 
    tail_bg, occ = T.build_tail_bg(src_q, mattes, tail)
    plate = G.build_plant_plate(src_q[121], T.norm_alpha(mattes[121]))
    leaf_plate = build_static_leaf_plate(src_q, mattes, tail_bg)
    # prefer the static leafy plate in the plant corner
    corner = T.sstep((Xg - 1088.0) / 10.0) * T.sstep((Yg - 560.0) / 10.0)
    tail_bg = tail_bg * (1.0 - corner[..., None]) + leaf_plate * corner[..., None]
    hook_occ = inherited_hook_occ(src_u, mattes)
    occ = np.maximum(occ, hook_occ)
    # widen occupancy so the inherited u-ex hook is fully cleared
    occ = np.clip(cv2.dilate((occ > 0.08).astype(np.uint8),
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))).astype(np.float32), 0, 1)
    occ = cv2.GaussianBlur(occ, (0, 0), 3.0)
    eyes = eye_mask()
    ears = ear_mask()
    tail['rest'] = rest
    ctx = dict(tail=tail, tail_bg=tail_bg, occ=occ, plate=plate, mic=mic,
               eyes=eyes, ears=ears)
    QC.mkdir(parents=True, exist_ok=True)

    if args.preview:
        idx = sorted(set(list(range(0, 241)) + list(range(420, 450))))
        out, areas, tips = process(src_u, mattes, ctx, preview=idx)
        dest = QC / 'preview'
        dest.mkdir(parents=True, exist_ok=True)
        tail_strip_2x(out, dest, every=6)
        meow_seq(out, dest)
        eye_seq(out, dest)
        before_after(out, src_u, dest)
        face_and_chest(out, src_u, dest)
        area_report(areas, dest, tips, rest[-1])
        T.mouth_qc(out, src_u, dest)
        print('preview written', dest, flush=True)
        return

    out, areas, tips = process(src_u, mattes, ctx)
    y0, y1, x0, x1 = TITLE
    bad = [i for i, (a, b) in enumerate(zip(src_u, out))
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
    shutil.copy2(Path(__file__), WORK / f'build_{VER.replace("-", "_")}.py')
    print('v41u-ex still', sha256(P / 'v41u-ex-test-25s.mp4'), flush=True)
    dest = QC / 'encoded-tmp'
    dest.mkdir(parents=True, exist_ok=True)
    tail_strip_2x(out, dest, every=6)
    meow_seq(out, dest)
    eye_seq(out, dest)
    before_after(out, src_u, dest)
    face_and_chest(out, src_u, dest)
    area_report(areas, dest, tips, rest[-1])
    T.mouth_qc(out, src_u, dest)


if __name__ == '__main__':
    main()
