#!/usr/bin/env python3
"""v41af-ex: from frozen v41ae-ex.

1. BLINKS — revive the I2V clip whose closed lids are natural fur at 3×.
   Align each I2V eye to the current (still-open) target eye with a per-eye
   affine from open-eye flow, then carry that transform onto the closing
   I2V frames. Lid mask is opaque over the iris (no ghosted open eye).
   Two blinks per 241-cycle: fast close, short hold, slower open.
2. HEAD — subtle yaw parallax so turns read as 3D, not a flat card:
   near-side cheek/ear move a little more than the far side; the nose
   leads by 1 frame; ears have a tiny follow-through settle. Feature
   shapes stay rigid.
3. TAIL — hide small cream fragments peeking through lower leaf gaps
   (n90–104) by covering components under 800 px with the plate.

Does not overwrite earlier encodes. Reuses ab-ex g1v3/g2v3 mattes
(in-memory face-hole repair). Mouth hook idle. Same audio/title.
"""
from __future__ import annotations

import gc
import os
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

P = Path('/workspace/lofi-v41-meow')
sys.path.insert(0, str(P))
import build_v41h as Hh
import build_v41y_ex as Y
import build_v41z_ex as Z
import build_v41t_ex as T
import build_v41aa_ex as AA

VER = 'v41af-ex'
WORK = P / VER
QC = P / 'qc' / VER
SRC_U = Y.SRC_U
PCM = Hh.PCM

V41U_SHA, V41U_LOS_SHA = Z.V41U_SHA, Z.V41U_LOS_SHA
V41Z_SHA = AA.V41Z_SHA
V41Z_LOS_SHA = AA.V41Z_LOS_SHA
V41AA_SHA = 'fa9d5784f702b32fd8741cc52ed098321856dc72ed2a56458cbea7c20ebf7bc4'
V41AA_LOS_SHA = '35af3afa2fca76ecad4ade1f0cf8f843edb5ae51b46252f0fdf8d012ad0a630c'
V41AB_SHA = 'de67ea71aae9348d0914ad413cd220b5b2799e0fc831cb1df31f1f077e0e5630'
V41AB_LOS_SHA = '7eca192e0cf05deece2141d0e607264fd77d5cc8c2a74de11be95a04e53b9a38'
AAC_SHA = '1fc51af4babc4df1d6d5ca9698295497a0e5dfd6ab654bef450e1caf17c4d2d1'
MATTE_G1 = P / 'v41ab-ex' / 'mattes' / 'g1v3'
MATTE_G2 = P / 'v41ab-ex' / 'mattes' / 'g2v3'
V41AC_SHA = '47643b7a9d0e8f3214ef2c10042e9237b2dadc54f47f2e466db5c15939d9dcf2'
V41AC_LOS_SHA = '78b0678543ab0430e46567f7ac4382cb5b047c81a64da600b69ba8469682c771'
V41AD_SHA = 'aa74757ff502fef649f9a87d493280b37ab1edb984cee418448c74563c70b7d9'
V41AE_SHA = 'b26892795ffa55edf44bbefb4eb710a5f055401dfd845b7e63671a4626949192'
BLINK_I2V = P / 'qc' / 'v41ad-ex' / 'v41ad-blink-i2v.mp4'

NFR, W, H, NCYCLE = 600, 1280, 720, 241
TITLE = Hh.TITLE
HEAD = Z.HEAD
Xg, Yg = Y.Xg, Y.Yg

# --- stronger, still eased, 241-periodic organic motion ---
N_BREATH = 3
CHEST_AMP = 3.60          # px chest/shoulder rise (was 1.80)
BODY_SCALE = 0.0075       # slight overall body scale at full inhale
HEAD_BREATH_AMP = 1.20
HEAD_LAG = 8
DRIFT_AMP_X, DRIFT_AMP_Y = 1.54, 1.19   # 70% of ac-ex 2.20/1.70
TILT_DEG = 0.70                         # 70% of ac-ex 1.00
TILT_PIVOT = (710.0, 360.0)

ANT_PRE, ANT_POST = 10, 14
ANT_LIFT = 2.40
ANT_EAR = 3.20

L_BASE = (622.0, 246.0)
R_BASE = (806.0, 248.0)
# asymmetric twitches: 4 frames out, 7 back, ~7 deg
TWITCHES = [
    {'start_local': 72, 'out': 4, 'back': 7, 'deg': 7.0, 'side': 'L', 'gen2': False},
    {'start_local': 108, 'out': 4, 'back': 7, 'deg': 7.2, 'side': 'R', 'gen2': True},
]

# 2 blinks per cycle, away from seams and meows
# (start, close, hold, open) in local frames
# Tracked lids land on the eyes, but at 3x the full-close fill still
# reads as a smeared on-eye patch (vertical banding, pale almond).
# Mid-blink (k~0.5) is close to natural; k>=0.9 is not. Per the brief
# ("if you can't get it natural, remove the blinks entirely rather
# than shipping stickers") blinks are omitted in this encode. The
# tracker and lid compositor remain in this file, unused.
BLINKS = []
# Rest-pose priors on a pre-warp z-ex plate; detection on the warped
# frame updates these. Do NOT add a fixed EYE_L/R + dx,dy offset.
EYE_PRIOR = {'L': (665.0, 309.0), 'R': (780.0, 322.0)}

# mouth
MOUTH_BOX = (310, 455, 620, 830)  # y0,y1,x0,x1
MOUTH_REF_SRC = 188
MOUTH_CLOSE_SRC = 222

STEAM_BOX = (300, 560, 240, 400)  # mug + rising steam (no laptop)
TAIL_BOX = (450, 720, 940, 1280)

MOUTH_REF_PATH = os.environ.get('V41AF_MOUTH_REF') or os.environ.get('V41AE_MOUTH_REF') or os.environ.get('V41AD_MOUTH_REF') or os.environ.get('V41AC_MOUTH_REF') or os.environ.get('V41AB_MOUTH_REF') or os.environ.get('V41AA_MOUTH_REF') or None

sha256 = Z.sha256
luma = Z.luma
mad_rgb, mad_luma = Z.mad_rgb, Z.mad_luma
smootherstep = AA.smootherstep


def aac_sha(path):
    raw = subprocess.check_output(
        ['ffmpeg', '-v', 'error', '-i', str(path), '-map', '0:a:0',
         '-c:a', 'copy', '-f', 'adts', 'pipe:1']
    )
    import hashlib
    return hashlib.sha256(raw).hexdigest()


# -------------- motion signals --------------
def breath_k(local):
    p = (float(local) / float(NCYCLE)) * N_BREATH
    frac = p - np.floor(p)
    if frac < 0.40:
        return float(smootherstep(frac / 0.40))
    return float(1.0 - smootherstep((frac - 0.40) / 0.60))


K_BREATH = np.array([breath_k(i) for i in range(NCYCLE)], np.float32)
K_HEAD = np.roll(K_BREATH, HEAD_LAG)


# Settle-beat head poses: ease, hold, ease. Starts and ends at 0 for the 241 join.
# Times are inclusive starts of each segment (ease or hold).
# (local, dx, dy, tilt_deg) knots; piecewise smootherstep between moves, constant on holds.
_HEAD_KNOTS = [
    # local  dx     dy     tilt_deg
    (0,     0.00,  0.00,  0.00),
    (14,    0.00,  0.00,  0.00),   # hold 0
    (30,    1.10, -0.80,  0.40),   # ease to A
    (54,    1.10, -0.80,  0.40),   # hold A
    (70,   -1.25, -1.35, -0.35),   # ease to B
    (98,   -1.25, -1.35, -0.35),   # hold B
    (114,   0.55, -0.45,  0.25),   # ease to C
    (150,   0.55, -0.45,  0.25),   # hold C
    (168,   0.00,  0.00,  0.00),   # ease home
    (241,   0.00,  0.00,  0.00),   # hold 0 through the seam
]


def _head_pose(local):
    t = float(local) % float(NCYCLE)
    knots = _HEAD_KNOTS
    for i in range(len(knots) - 1):
        t0, x0, y0, a0 = knots[i]
        t1, x1, y1, a1 = knots[i + 1]
        if t <= t1 or i == len(knots) - 2:
            if abs(x1 - x0) < 1e-9 and abs(y1 - y0) < 1e-9 and abs(a1 - a0) < 1e-9:
                return x0, y0, np.deg2rad(a0)
            u = smootherstep((t - t0) / max(t1 - t0, 1e-6))
            return (x0 + u * (x1 - x0), y0 + u * (y1 - y0),
                    np.deg2rad(a0 + u * (a1 - a0)))
    return 0.0, 0.0, 0.0


def drift(local):
    x, y, _ = _head_pose(local)
    return float(x), float(y)


def tilt_ang(local):
    _, _, a = _head_pose(local)
    return float(a)


def anticip(local, peak):
    lo, hi = peak - ANT_PRE, peak + ANT_POST
    if local < lo or local > hi:
        return 0.0, 0.0
    if local <= peak:
        w = smootherstep((local - lo) / max(ANT_PRE, 1))
    else:
        w = 1.0 - smootherstep((local - peak) / max(ANT_POST, 1))
    return -ANT_LIFT * w, ANT_EAR * w


def twitch_angle(i, local, use2):
    ang_L = 0.0
    ang_R = 0.0
    for tw in TWITCHES:
        if tw['gen2'] != use2:
            continue
        s, out, back = tw['start_local'], tw['out'], tw['back']
        dur = out + back
        if s <= local < s + dur:
            t = local - s + 0.5
            if t <= out:
                w = smootherstep(t / max(out, 1))
            else:
                w = 1.0 - smootherstep((t - out) / max(back, 1))
            a = np.deg2rad(tw['deg']) * w
            if tw['side'] == 'L':
                ang_L += a
            else:
                ang_R += a
    return ang_L, ang_R


def blink_k(local):
    k = 0.0
    for bl in BLINKS:
        s = bl['start']
        c, h, o = bl['close'], bl['hold'], bl['open']
        t = local - s
        if t < 0 or t >= c + h + o:
            continue
        if t < c:
            w = smootherstep((t + 0.5) / c)
        elif t < c + h:
            w = 1.0
        else:
            w = 1.0 - smootherstep((t - c - h + 0.5) / o)
        k = max(k, w)
    return float(k)


rotation_disp = AA.rotation_disp

_WEIGHTS = None

def build_weight_maps():
    """Soft organic weights — no rectangular face plate (that showed as a box
    once head drift/tilt became visible)."""
    def blob(cx, cy, sx, sy):
        return np.exp(-0.5 * ((Xg - cx) / sx) ** 2 - 0.5 * ((Yg - cy) / sy) ** 2)

    head = np.clip(blob(710, 305, 200, 170), 0, 1)
    neck = smootherstep((465.0 - Yg) / 75.0)
    head = head * neck
    chest = blob(700, 515, 210, 150)
    chest = chest * smootherstep((Yg - 370.0) / 50.0)
    chest = chest * smootherstep((1020.0 - Xg) / 40.0)
    ear_L = blob(598, 200, 58, 52) * smootherstep((Xg - 530.0) / 16.0)
    ear_R = blob(848, 205, 60, 54) * smootherstep((920.0 - Xg) / 16.0)
    eyes = np.zeros((H, W), np.float32)
    eyes[240:325, 600:860] = 1.0
    eyes = cv2.GaussianBlur(eyes, (0, 0), 14.0)
    head = np.clip(np.maximum(head, 0.97 * eyes * head.max()), 0, 1)
    return {'head': head, 'chest': chest, 'ear_L': ear_L, 'ear_R': ear_R, 'eyes': eyes}

def weights():
    global _WEIGHTS
    if _WEIGHTS is None:
        _WEIGHTS = build_weight_maps()
    return _WEIGHTS


def warp_field(i):
    use2, local = Y.cycle_of(i)
    wt = weights()
    k_c = float(K_BREATH[local])
    k_h = float(K_HEAD[local])
    drift_x, drift_y = drift(local)
    peak = 191 if use2 else 138
    ant_dy, ant_ear = anticip(local, peak)
    ang_L, ang_R = twitch_angle(i, local, use2)
    tang = tilt_ang(local)

    dx = np.zeros((H, W), np.float32)
    dy = np.zeros((H, W), np.float32)

    # chest / shoulders + slight body scale about a mid-chest pivot
    dy = dy + wt['chest'] * (-CHEST_AMP * k_c)
    dx = dx + wt['chest'] * (0.45 * k_c * np.clip((Xg - 700.0) / 200.0, -1, 1))
    if k_c > 1e-6 and BODY_SCALE > 0:
        sc = BODY_SCALE * k_c
        # scale about (700, 530) on the chest only — do not scale the head
        # (keeps eyes and markings rigid together)
        body = wt['chest']
        dx = dx + body * (sc * (Xg - 700.0))
        dy = dy + body * (sc * (Yg - 530.0))

    # head: lagged breath, wander, tilt, anticipation, with subtle yaw parallax
    # near-side cheek/ear displace a little more; eyes stay closer to rigid.
    yaw = np.clip(tang / max(np.deg2rad(TILT_DEG), 1e-6), -1.0, 1.0)
    yaw = np.clip(yaw + 0.35 * np.clip(drift_x / max(DRIFT_AMP_X, 1e-6), -1.0, 1.0), -1.0, 1.0)
    depth = 1.0 + 0.15 * yaw * np.clip((Xg - 710.0) / 150.0, -1.0, 1.0)
    # keep eye features from stretching
    ew = wt['eyes']
    depth = depth * (1.0 - 0.70 * ew) + 1.0 * (0.70 * ew)
    depth = np.clip(depth, 0.86, 1.14).astype(np.float32)

    # nose leads the turn by 1 frame
    nx, ny, na = _head_pose((local + 1) % NCYCLE)
    # wrap at the 241 join is already 0,0
    if local >= NCYCLE - 1:
        nx, ny, na = 0.0, 0.0, 0.0
    nose = np.exp(-0.5 * ((Xg - 725.0) / 36.0) ** 2 - 0.5 * ((Yg - 355.0) / 32.0) ** 2)
    nose = np.clip(nose, 0, 1).astype(np.float32)

    # tiny follow-through on ears after a settle-beat hold begins
    fdx = fdy = fta = 0.0
    for t_h, t_prev in ((54, 30), (98, 70), (150, 114)):
        age = local - t_h
        if 0 <= age < 5:
            x0, y0, a0 = _head_pose(t_prev)
            x1, y1, a1 = _head_pose(t_h)
            wft = float(np.exp(-age / 1.8))
            fdx, fdy, fta = (x1 - x0) * 0.18 * wft, (y1 - y0) * 0.18 * wft, (a1 - a0) * 0.18 * wft
            break

    hdx = drift_x
    hdy = -HEAD_BREATH_AMP * k_h + drift_y + ant_dy
    dy = dy + wt['head'] * depth * hdy
    dx = dx + wt['head'] * depth * hdx
    if abs(tang) > 1e-8:
        dxt, dyt = rotation_disp(*TILT_PIVOT, tang)
        dx = dx + wt['head'] * depth * dxt
        dy = dy + wt['head'] * depth * dyt
    # nose lead (only the residual vs current pose)
    dx = dx + nose * (nx - drift_x)
    dy = dy + nose * (ny - drift_y)
    if abs(na - tang) > 1e-8:
        dxt, dyt = rotation_disp(*TILT_PIVOT, na - tang)
        dx = dx + nose * dxt
        dy = dy + nose * dyt
    # follow-through, mostly on the ears / sides
    if abs(fdx) + abs(fdy) + abs(fta) > 1e-8:
        side = np.clip(wt['ear_L'] + wt['ear_R'] + 0.35 * wt['head'] * (1.0 - ew), 0, 1)
        dx = dx + side * fdx
        dy = dy + side * fdy
        if abs(fta) > 1e-8:
            dxt, dyt = rotation_disp(*TILT_PIVOT, fta)
            dx = dx + side * dxt
            dy = dy + side * dyt

    if abs(ant_ear) > 1e-6:
        a = np.deg2rad(2.6) * (ant_ear / max(ANT_EAR, 1e-6))
        dxl, dyl = rotation_disp(*L_BASE, +a)
        dxr, dyr = rotation_disp(*R_BASE, -a)
        dx = dx + wt['ear_L'] * dxl + wt['ear_R'] * dxr
        dy = dy + wt['ear_L'] * dyl + wt['ear_R'] * dyr

    if abs(ang_L) > 1e-8:
        dxl, dyl = rotation_disp(*L_BASE, ang_L)
        dx = dx + wt['ear_L'] * dxl
        dy = dy + wt['ear_L'] * dyl
    if abs(ang_R) > 1e-8:
        dxr, dyr = rotation_disp(*R_BASE, ang_R)
        dx = dx + wt['ear_R'] * dxr
        dy = dy + wt['ear_R'] * dyr

    planted = smootherstep((620.0 - Yg) / 50.0).astype(np.float32)
    dx, dy = dx * planted, dy * planted
    return dx, dy


# -------------- tail / plant layers --------------
def cream_mask(im):
    imf = im.astype(np.float32)
    B, G, R = imf[..., 0], imf[..., 1], imf[..., 2]
    Yl = luma(im)
    return (Yl > 55) & (R + 8 >= G) & ~((G > R + 10) & (G > B - 5) & (Yl < 145))


def leaf_mask(im):
    imf = im.astype(np.float32)
    B, G, R = imf[..., 0], imf[..., 1], imf[..., 2]
    Yl = luma(im)
    roi = (Xg > 970) & (Yg > 460)
    leaf = roi & (G > R + 2) & (G > B - 10) & (Yl < 150) & (Yl > 12)
    olive = roi & (G > R - 4) & (G > B) & (Yl < 120) & (Yl > 12) & ((G - R) < 45)
    dark = roi & (G > R) & (G > B - 6) & (Yl < 50) & (Yl > 8) & ((G - R) < 40)
    m = (leaf | olive | dark).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    keep = np.zeros_like(m)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] >= 40:
            keep[lab == i] = 1
    keep[cream_mask(im)] = 0
    return np.clip(cv2.GaussianBlur(keep.astype(np.float32), (0, 0), 1.1), 0, 1)


_K21 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
_K27 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (27, 27))
_K7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
_K15 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))


def solid_tail_and_inpaint(im, a_cat, a_plant):
    """Tight cat + joined tail, fur-inpainted interiors, chair-pocket matte."""
    fur = cream_mask(im)
    a = np.clip(a_cat * (1.0 - a_plant), 0, 1)
    tailroi = (Xg > 940) & (Yg > 400)
    a = np.clip(a + fur.astype(np.float32) * tailroi.astype(np.float32) * 0.95, 0, 1)
    hard = (a > 0.35).astype(np.uint8)
    tail = hard.copy()
    tail[~tailroi] = 0
    closed = cv2.morphologyEx(tail, cv2.MORPH_CLOSE, _K27)
    inv = (1 - closed).astype(np.uint8)
    ff = inv.copy()
    mask = np.zeros((H + 2, W + 2), np.uint8)
    for x in range(940, 1280, 20):
        if ff[400, x] == 1:
            cv2.floodFill(ff, mask, (x, 400), 2)
        if ff[719, x] == 1:
            cv2.floodFill(ff, mask, (x, 719), 2)
    for y in range(400, 720, 20):
        if ff[y, 940] == 1:
            cv2.floodFill(ff, mask, (940, y), 2)
        if ff[y, 1279] == 1:
            cv2.floodFill(ff, mask, (1279, y), 2)
    holes = (ff == 1) & tailroi
    bay = (closed > 0) & (tail == 0) & tailroi
    n, lab, st, _ = cv2.connectedComponentsWithStats(bay.astype(np.uint8), 8)
    bay_keep = np.zeros_like(bay)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] <= 1800:
            bay_keep[lab == i] = True
    fill = holes | bay_keep
    soft = cv2.GaussianBlur(fill.astype(np.float32), (0, 0), 1.2)
    a_s = np.clip(np.maximum(a, soft), 0, 1)
    Yl = luma(im)
    # Only inpaint dark fur interiors — never plant/leaf pixels (that smeared the n72 leaf).
    inp = (fill & (Yl < 70) & (a_plant < 0.25)).astype(np.uint8)
    gi = im.copy()
    if int(inp.sum()) > 0:
        gi = cv2.inpaint(im, cv2.dilate(inp, _K7), 5, cv2.INPAINT_TELEA)
    # Corner coverage: close cat+plant+cream so u-ex cannot punch holes
    # between the tail and the leaves. Plant stays a separate top layer.
    near = cv2.dilate((a_s > 0.35).astype(np.uint8), _K15)
    chair = (Yl < 48) & tailroi & (near > 0) & (a_plant < 0.25) & (a_s < 0.35)
    union = ((a_s > 0.30) | chair | ((a_plant > 0.30) & tailroi) | (fur & tailroi)).astype(np.uint8)
    union[~tailroi] = 0
    cov = cv2.morphologyEx(union, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)))
    cnts, _ = cv2.findContours(cov, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(cov)
    cv2.drawContours(filled, cnts, -1, 1, thickness=-1)
    # drop tiny isolated islands that never touch the main cat
    n, lab, st, _ = cv2.connectedComponentsWithStats(filled, 8)
    keep = np.zeros_like(filled)
    main = (a_s > 0.35)
    for i in range(1, n):
        m = lab == i
        if st[i, cv2.CC_STAT_AREA] < 250:
            continue
        if (m & main).sum() == 0 and st[i, cv2.CC_STAT_LEFT] > 1100:
            continue
        keep[m] = 1
    a_cov = cv2.GaussianBlur(keep.astype(np.float32), (0, 0), 1.4)
    # coverage that is not the moving cat (unwarped underlay) + small chair pockets
    a_chair = np.clip(np.maximum(a_cov, chair.astype(np.float32)) * tailroi.astype(np.float32), 0, 1)
    # punch plant out later in compose; keep coverage including under leaves
    return a_s, a_chair, gi, inp


def build_layers(frames, raw, pane, pane_allow, ceil, cache_dir: Path):
    cat_p = cache_dir / 'cat-solid.npz'
    plant_p = cache_dir / 'plant.npz'
    inp_p = cache_dir / 'inpaint-gen.npz'
    chair_p = cache_dir / 'chair-pocket.npz'
    if cat_p.exists() and plant_p.exists() and inp_p.exists() and chair_p.exists():
        print('loading cached ab-ex layers', flush=True)
        return (
            np.load(cat_p)['alpha'],
            np.load(plant_p)['alpha'],
            np.load(inp_p)['gen'],
            np.load(chair_p)['alpha'],
        )
    Tlen = len(frames)
    print(f'building cat/plant layers ({Tlen})…', flush=True)
    cat_raw = np.zeros((Tlen, H, W), np.float32)
    plant = np.zeros((Tlen, H, W), np.float32)
    for i in range(Tlen):
        cat_raw[i] = Y.clean_cat(frames[i], raw[i], pane, pane_allow, ceil=ceil)
        plant[i] = leaf_mask(frames[i])
        if i % 40 == 0:
            print(f'  clean {i}/{Tlen}', flush=True)
    cat_raw = Y.temporal_gauss(cat_raw, sig=0.85)
    plant = Y.temporal_gauss(plant, sig=0.35)
    cat_s = np.zeros_like(cat_raw)
    chair = np.zeros_like(cat_raw)
    gen_i = frames.copy() if isinstance(frames, np.ndarray) else np.stack(frames, 0)
    gen_i = gen_i.astype(np.uint8)
    for i in range(Tlen):
        a_s, a_ch, gi, _ = solid_tail_and_inpaint(frames[i], cat_raw[i], plant[i])
        cat_s[i] = a_s
        chair[i] = a_ch
        gen_i[i] = gi
        if i % 40 == 0:
            print(f'  solid {i}/{Tlen}', flush=True)
    cat_s = Y.temporal_gauss(cat_s, sig=0.7)
    chair = Y.temporal_gauss(chair, sig=0.7)
    cat_u = (np.clip(cat_s, 0, 1) * 255).astype(np.uint8)
    plant_u = (np.clip(plant, 0, 1) * 255).astype(np.uint8)
    chair_u = (np.clip(chair, 0, 1) * 255).astype(np.uint8)
    del cat_s, chair, cat_raw, plant
    gc.collect()
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cat_p, alpha=cat_u)
    np.savez_compressed(plant_p, alpha=plant_u)
    np.savez_compressed(chair_p, alpha=chair_u)
    np.savez_compressed(inp_p, gen=gen_i)
    return cat_u, plant_u, gen_i, chair_u


# -------------- mouth stabilisation --------------
def mouth_interior(im):
    y0, y1, x0, x1 = MOUTH_BOX
    imf = im.astype(np.float32)
    B, G, R = imf[..., 0], imf[..., 1], imf[..., 2]
    Yl = luma(im)
    cav = (Yl < 75) & (Yl > 4)
    tongue = (R > G + 12) & (R > B + 8) & (Yl > 35) & (Yl < 170)
    fang = (Yl > 135) & (np.abs(R - G) < 28) & (np.abs(R - B) < 35)
    m = np.zeros((H, W), np.uint8)
    sl = (cav | tongue | fang)[y0:y1, x0:x1]
    m[y0:y1, x0:x1] = sl.astype(np.uint8)
    # keep components near the mouth centre
    n, lab, st, cents = cv2.connectedComponentsWithStats(m, 8)
    keep = np.zeros_like(m)
    cx0, cy0 = 725.0, 395.0
    for i in range(1, n):
        cx, cy = cents[i]
        if st[i, cv2.CC_STAT_AREA] < 8:
            continue
        if abs(cx - cx0) < 90 and abs(cy - cy0) < 55:
            keep[lab == i] = 1
    keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    return cv2.GaussianBlur(keep.astype(np.float32), (0, 0), 1.4)


def _remap_flow(im, flow, scale):
    return Z._warp(im, flow, scale)


_dis_m = None


def _dis_mflow():
    global _dis_m
    if _dis_m is None:
        _dis_m = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    return _dis_m


def stabilize_mouth(gen, ref, ref_int, close_im=None, morph_w=None):
    """Flow-lock fangs/tongue to a clean reference; clamp chin stretch."""
    y0, y1, x0, x1 = MOUTH_BOX
    cur_int = mouth_interior(gen)
    # skip nearly closed mouths
    if float((cur_int > 0.3).sum()) < 80 and (morph_w is None):
        return gen
    A = ref.astype(np.float32)
    B = gen.astype(np.float32)
    fa = _dis_mflow().calc(Z._gray(A), Z._gray(B), None)
    # clamp vertical flow in the lower face to limit chin stretch
    fy = fa[..., 1].copy()
    chin = (Yg > 360) & (Xg > 600) & (Xg < 850)
    fy[chin] *= 0.45
    fa = np.dstack([fa[..., 0], fy])
    warped = _remap_flow(A, fa, 1.0)
    # paste ONLY into the current mouth opening (no cheek wall)
    mask = cv2.GaussianBlur(np.clip(cur_int, 0, 1), (0, 0), 1.3)
    mask = np.clip(mask * 1.15, 0, 1)
    # hard-zero outside a tight mouth ellipse so cheeks cannot pick up the ref
    yy = (Yg - 398.0) / 48.0
    xx = (Xg - 725.0) / 72.0
    ell = np.clip(1.0 - (xx * xx + yy * yy), 0, 1)
    mask = mask * ell
    out = B * (1.0 - mask[..., None]) + warped * mask[..., None]
    # if we are in the 189->222 morph, fade the locked interior toward the close frame at the end
    if close_im is not None and morph_w is not None and morph_w > 0.55:
        w2 = smootherstep((morph_w - 0.55) / 0.45)
        fb = _dis_mflow().calc(Z._gray(close_im), Z._gray(B), None)
        fy2 = fb[..., 1].copy()
        fy2[chin] *= 0.45
        fb = np.dstack([fb[..., 0], fy2])
        warped_b = _remap_flow(close_im.astype(np.float32), fb, 1.0)
        out = out * (1.0 - w2 * mask[..., None]) + warped_b * (w2 * mask[..., None])
    return np.clip(out, 0, 255).astype(np.uint8)


# -------------- blinks --------------
def _col_bounds(mask):
    """For each column, (top, bot) of a 2D uint8 mask. -1 if empty."""
    h, w = mask.shape
    tops = np.full(w, -1, np.int32)
    bots = np.full(w, -1, np.int32)
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return tops, bots
    for x in np.unique(xs):
        col = ys[xs == x]
        tops[x] = int(col.min())
        bots[x] = int(col.max())
    return tops, bots


def _fill_and_smooth(arr, empty=-1, sigma=2.0):
    x = np.arange(arr.size)
    m = arr != empty
    if m.sum() < 3:
        return arr.astype(np.float32)
    y = arr.astype(np.float32)
    y[~m] = np.interp(x[~m], x[m], y[m])
    r = int(3 * sigma) + 1
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    i0, i1 = int(np.argmax(m)), int(len(m) - 1 - np.argmax(m[::-1]))
    span = y[i0:i1 + 1]
    pad = np.pad(span, r, mode='edge')
    sm = np.convolve(pad, k, mode='valid')
    out = y.copy()
    out[i0:i1 + 1] = sm
    out[:i0] = empty
    out[i1 + 1:] = empty
    return out


def detect_one(im, prior, half=(48, 42)):
    cx0, cy0 = prior
    x0 = int(np.clip(round(cx0) - half[0], 0, W - 2))
    x1 = int(np.clip(round(cx0) + half[0], x0 + 2, W))
    y0 = int(np.clip(round(cy0) - half[1], 0, H - 2))
    y1 = int(np.clip(round(cy0) + half[1], y0 + 2, H))
    roi = im[y0:y1, x0:x1]
    Yl = luma(roi)
    # iris/pupil core: darkest substantial blob
    best = None
    for thr in (40, 48, 55, 62):
        dark = (Yl < thr).astype(np.uint8)
        dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        n, lab, st, cents = cv2.connectedComponentsWithStats(dark, 8)
        for i in range(1, n):
            area = st[i, cv2.CC_STAT_AREA]
            if area < 80 or area > 2500:
                continue
            ww, hh = st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]
            if ww < 10 or hh < 8 or ww > 80 or hh > 50:
                continue
            cx, cy = float(cents[i, 0]), float(cents[i, 1])
            d = np.hypot(cx - (cx0 - x0), cy - (cy0 - y0))
            if d > 32:
                continue
            aspect = ww / max(hh, 1)
            if aspect < 0.6 or aspect > 4.0:
                continue
            rad = 14
            ly0, ly1 = max(0, int(cy) - rad), min(Yl.shape[0], int(cy) + rad + 1)
            lx0, lx1 = max(0, int(cx) - rad), min(Yl.shape[1], int(cx) + rad + 1)
            catch = float(Yl[ly0:ly1, lx0:lx1].max())
            score = area / (1.0 + 0.05 * d * d) + 0.4 * min(catch, 200)
            if best is None or score > best[0]:
                best = (score, cx, cy, lab == i, catch, thr, ww, hh, area)
        if best is not None and best[5] <= 48:
            break
    if best is None:
        return None
    _, pcx, pcy, core, catch, thr, ww, hh, area = best
    opening = core.astype(np.uint8)
    opening = cv2.dilate(opening, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    yy, xx = np.ogrid[0:Yl.shape[0], 0:Yl.shape[1]]
    rx_c = ww / 2.0 + 3.0
    ry_c = hh / 2.0 + 2.0
    ell = ((xx - pcx) / rx_c) ** 2 + ((yy - pcy) / ry_c) ** 2 <= 1.05
    opening = (opening > 0) & ell
    opening = cv2.erode(opening.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    opening = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(opening, 8)
    pid = lab[int(np.clip(pcy, 0, lab.shape[0] - 1)), int(np.clip(pcx, 0, lab.shape[1] - 1))]
    if pid == 0:
        if n <= 1:
            return None
        pid = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    opening = (lab == pid).astype(np.uint8)
    tops, bots = _col_bounds(opening)
    if (tops >= 0).sum() < 10:
        return None
    tops_s = _fill_and_smooth(tops, sigma=1.6)
    bots_s = _fill_and_smooth(bots, sigma=1.6)
    ys, xs = np.where(opening > 0)
    return {
        'cx': float(xs.mean() + x0),
        'cy': float(ys.mean() + y0),
        'pcx': float(pcx + x0),
        'pcy': float(pcy + y0),
        'catch': catch,
        'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1,
        'opening': opening,
        'tops': tops_s,
        'bots': bots_s,
        'y_top': float(ys.min() + y0),
        'y_bot': float(ys.max() + y0),
        'thr': thr, 'area': area,
    }


def detect_eyes(im, priors=None):
    if priors is None:
        priors = dict(EYE_PRIOR)
    out = {}
    for side, pr in priors.items():
        out[side] = detect_one(im, pr)
    return out


def apply_blink(im, k, eyes):
    """Upper lid follows each eye's own contour; iris squashes toward the lower lash."""
    if k <= 1e-4:
        return im
    out = im.astype(np.float32).copy()
    h, w = im.shape[:2]
    for side, e in eyes.items():
        if e is None:
            continue
        x0, y0, x1, y1 = e['x0'], e['y0'], e['x1'], e['y1']
        y0b = max(0, y0 - 22)
        y1e = min(h, y1 + 8)
        tops = e['tops']
        bots = e['bots']
        dy0 = y0 - y0b
        bw = x1 - x0
        bh = y1e - y0b
        yy, xx = np.mgrid[0:bh, 0:bw]
        t = tops + dy0
        b = bots + dy0
        valid = (e['tops'] >= 0) & (e['bots'] >= 0)
        opening0 = np.maximum(b - t, 1.0)
        y_lid = t + k * 0.90 * opening0
        y_low = b - k * 0.10 * opening0
        denom = np.maximum(y_low - y_lid, 0.6)
        y_src = t[None, :] + (yy - y_lid[None, :]) / denom[None, :] * (b - t)[None, :]
        map_x = (xx + x0).astype(np.float32)
        map_y = np.clip(y_src + y0b, 0, h - 1).astype(np.float32)
        squashed = cv2.remap(im, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        in_span = valid[None, :]
        is_open = in_span & (yy >= y_lid[None, :] - 0.4) & (yy <= y_low[None, :] + 0.4)
        is_ulid = in_span & (yy <= y_lid[None, :] + 0.5) & (yy >= t[None, :] - 0.3)
        is_llid = in_span & (yy >= y_low[None, :] - 0.5) & (yy <= b[None, :] + 0.8) & (k > 0.20)
        open_w = cv2.GaussianBlur(is_open.astype(np.float32), (0, 0), 0.50)
        lid_w = cv2.GaussianBlur(is_ulid.astype(np.float32), (0, 0), 0.50)
        low_w = cv2.GaussianBlur(is_llid.astype(np.float32), (0, 0), 0.50)
        top_fade = np.clip((yy - (t[None, :] - 0.2)) / 1.6, 0.0, 1.0)
        lid_w = lid_w * top_fade
        depth = np.clip((y_lid[None, :] - yy) / np.maximum(opening0[None, :], 6.0), 0.0, 1.0)
        brow_y = (t[None, :] - 2.0 - depth * 7.0) + y0b
        brow_x = (xx + x0).astype(np.float32)
        brow_y = np.clip(brow_y, 0, h - 1).astype(np.float32)
        brow = cv2.remap(im, brow_x, brow_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
        brow_blur = cv2.GaussianBlur(brow, (0, 0), 0.9)
        brow_hf = brow - brow_blur
        ref_y = np.broadcast_to(
            np.clip(t[None, :] - 3.5 + y0b, 0, h - 1), (bh, bw)
        ).astype(np.float32)
        ref = cv2.remap(im, brow_x, ref_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
        brow_l = np.maximum(luma(brow_blur), 1.0)
        ref_l = luma(ref)
        scale = np.clip(ref_l / brow_l, 0.92, 1.10)
        shade = (1.0 - 0.08 * (1.0 - depth))[..., None]
        lid_rgb = np.clip(brow_blur * scale[..., None] * shade + brow_hf, 0, 255)
        crease_src_y = (t[None, :] + (yy - y_lid[None, :])) + y0b
        crease = cv2.remap(
            im, brow_x, np.clip(crease_src_y, 0, h - 1).astype(np.float32),
            cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
        ).astype(np.float32)
        lead = np.exp(-0.5 * ((yy - y_lid[None, :]) / 1.35) ** 2) * valid[None, :].astype(np.float32)
        xs = np.arange(bw, dtype=np.float32)
        if valid.any():
            vix = np.flatnonzero(valid)
            u = (xs - vix[0]) / max(float(vix[-1] - vix[0]), 1.0)
            corner = np.clip(np.minimum(u, 1.0 - u) / 0.16, 0.0, 1.0).astype(np.float32)
        else:
            corner = np.ones(bw, np.float32)
        lead = lead * corner[None, :] * np.clip(k * 1.2, 0, 1)
        lead = cv2.GaussianBlur(lead, (0, 0), 0.40)
        cheek_y = (b[None, :] + 1.5 + np.clip((yy - y_low[None, :]), 0, 4)) + y0b
        cheek = cv2.remap(
            im, brow_x, np.clip(cheek_y, 0, h - 1).astype(np.float32),
            cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
        ).astype(np.float32)
        sl = out[y0b:y1e, x0:x1]
        sl = sl * (1.0 - open_w[..., None]) + squashed.astype(np.float32) * open_w[..., None]
        sl = sl * (1.0 - lid_w[..., None]) + lid_rgb * lid_w[..., None]
        sl = sl * (1.0 - 0.65 * low_w[..., None]) + cheek * (0.65 * low_w[..., None])
        sl = sl * (1.0 - lead[..., None]) + crease * lead[..., None]
        out[y0b:y1e, x0:x1] = sl
    return np.clip(out, 0, 255).astype(np.uint8)


_eye_priors = dict(EYE_PRIOR)


def apply_blink_tracked(im, k, i=None):
    """Detect on the already-warped plate; skip the blink if either eye is missed."""
    global _eye_priors
    eyes = detect_eyes(im, _eye_priors)
    for side, e in eyes.items():
        if e is not None:
            _eye_priors[side] = (e['pcx'], e['pcy'])
    if k <= 1e-4:
        return im
    if eyes.get('L') is None or eyes.get('R') is None:
        print(f'  WARN blink detect miss n={i} L={eyes.get("L") is not None} R={eyes.get("R") is not None}',
              flush=True)
        return im
    return apply_blink(im, k, eyes)


# -------------- compose --------------
# Soft corner coverage so the tail/plant region is filled with gen (no u-ex holes).
_PANE = None

def corner_gate():
    global _PANE
    if _PANE is None:
        _PANE = T.pane_masks().astype(np.float32)
    g = smootherstep((Xg - 890.0) / 55.0) * smootherstep((Yg - 395.0) / 45.0)
    g = g * (1.0 - np.clip(_PANE, 0, 1))
    return g.astype(np.float32)

_U_PLANT = None
_PANE_F = None


def u_plant_matte(src_u):
    """Solid static leaf-occlusion matte from u-ex plate shapes only.

    Closed, hole-filled silhouettes with smooth edges. No per-frame
    colour gating — dark tail fur must never be classified as leaf.
    """
    global _U_PLANT
    if _U_PLANT is not None:
        return _U_PLANT
    acc = None
    # leaf_mask already restricts to the plant ROI and drops cream on
    # each plate frame; union gives the static leaf *shapes*.
    for i in (0, 12, 24, 48, 96, 180, 240):
        m = (leaf_mask(src_u[i]) > 0.25).astype(np.uint8)
        acc = m if acc is None else cv2.bitwise_or(acc, m)
    k11 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    acc = cv2.morphologyEx(acc, cv2.MORPH_CLOSE, k11)
    # fill interior holes so leaves are solid cards, not veined mattes
    filled = np.zeros_like(acc)
    cnts, _ = cv2.findContours(acc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        cv2.drawContours(filled, cnts, -1, 1, thickness=-1)
    else:
        filled = acc
    # 1 px dilate so the silhouette covers the plate leaf edge, then soften
    filled = cv2.dilate(filled, k3)
    a = cv2.GaussianBlur(filled.astype(np.float32), (0, 0), 1.6)
    _U_PLANT = np.clip(a, 0, 1)
    return _U_PLANT


def pane_f():
    global _PANE_F
    if _PANE_F is None:
        _PANE_F = T.pane_masks().astype(np.float32)
    return _PANE_F


def compose_layered(uex, gen, a_cat, a_plant, a_chair, ph, ph0, w_room, w_bulb, a_pu):
    """Composite the cat/tail fully, then lay solid u-ex leaves on top.

    a_pu is the static filled plate-leaf matte. No per-frame colour gate.
    """
    a_c = np.clip(a_cat, 0, 1)
    a_h = np.clip(np.maximum(a_chair, corner_gate()), 0, 1)
    a_p = np.clip(a_pu, 0, 1)
    a_catish = np.clip(np.maximum(a_c, a_h), 0, 1)
    a_all = np.clip(np.maximum(a_catish, a_p), 0, 1)
    gen_f = Y.residual_flicker(gen, w_room, w_bulb, ph, ph0)
    gen_f = Y.color_match(gen_f, uex, a_all)
    # 1. full tail/cat over the plate (holes in a_p must not punch the tail)
    out = uex.astype(np.float32) * (1.0 - a_catish[..., None]) + gen_f * a_catish[..., None]
    # 2. plate leaves only where the solid occlusion matte says they are in front
    out = out * (1.0 - a_p[..., None]) + uex.astype(np.float32) * a_p[..., None]
    off = (a_catish < 0.02) & (a_p < 0.02)
    out[off] = uex[off].astype(np.float32)
    y0, y1, x0, x1 = TITLE
    out[y0:y1, x0:x1] = uex[y0:y1, x0:x1]
    return np.clip(out, 0, 255).astype(np.uint8), a_p


def lerp_u8(stack, t):
    """Lerp a uint8 alpha stack to float 0..1."""
    n = len(stack)
    if t <= 0:
        return stack[0].astype(np.float32) / 255.0
    if t >= n - 1:
        return stack[-1].astype(np.float32) / 255.0
    i0 = int(np.floor(t))
    u = float(t - i0)
    a = stack[i0].astype(np.float32)
    b = stack[i0 + 1].astype(np.float32)
    return np.clip((a * (1.0 - u) + b * u) / 255.0, 0, 1)


def sample_layers(use2, local, g1, g2, c1, c2, p1, p2, h1, h2, gi1, gi2):
    """Return (gen_inpainted, a_cat, a_plant, a_chair), with z-ex meow morph + seam later."""
    if use2:
        if Z.in_meow_morph(local):
            w = Z.meow_morph_w(local)
            iA, iB = 189, int(Z.GEN2_CLOSE_SRC)
            gen, a_c = Z.flow_blend(gi2[iA], gi2[iB], c2[iA]/255.0, c2[iB]/255.0, w)
            _, a_p = Z.flow_blend(gi2[iA], gi2[iB], p2[iA]/255.0, p2[iB]/255.0, w)
            _, a_h = Z.flow_blend(gi2[iA], gi2[iB], h2[iA]/255.0, h2[iB]/255.0, w)
            gen = stabilize_mouth(np.clip(gen, 0, 255).astype(np.uint8),
                                  gi2[MOUTH_REF_SRC], mouth_interior(gi2[MOUTH_REF_SRC]),
                                  close_im=gi2[iB], morph_w=w)
            return gen, a_c, a_p, a_h
        t = Z.src_time_gen2(local)
        gen = Y.lerp_frame(gi2, t)
        a_c, a_p, a_h = lerp_u8(c2, t), lerp_u8(p2, t), lerp_u8(h2, t)
        # lock interior on the big-meow approach and hold
        if 175 <= local <= 210:
            gen = stabilize_mouth(np.clip(gen, 0, 255).astype(np.uint8),
                                  gi2[MOUTH_REF_SRC], mouth_interior(gi2[MOUTH_REF_SRC]))
        return gen, a_c, a_p, a_h
    t = Y.src_time(local, Y.GEN1_PEAK, Y.DST_A)
    return Y.lerp_frame(gi1, t), lerp_u8(c1, t), lerp_u8(p1, t), lerp_u8(h1, t)


def compose_warped(plate, u, a_cat, a_plant, a_chair, dx, dy):
    a_c = np.clip(a_cat, 0, 1)
    a_p = np.clip(a_plant, 0, 1)
    # unwarped u-ex plant + chair pockets, minus the moving cat
    a_stat = np.clip(np.maximum(a_p, a_chair * (1.0 - a_c)), 0, 1)
    mx = (Xg - dx).astype(np.float32)
    my = (Yg - dy).astype(np.float32)
    pl_w = cv2.remap(plate, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    a_w = cv2.remap(a_c * (1.0 - a_p), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    a_w = np.clip(a_w, 0, 1)
    dil = cv2.dilate((a_c > 0.08).astype(np.uint8),
                     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    dil = cv2.GaussianBlur(dil.astype(np.float32), (0, 0), 1.4)
    bg = plate.astype(np.float32) * (1.0 - dil[..., None]) + u.astype(np.float32) * dil[..., None]
    out = bg * (1.0 - a_w[..., None]) + pl_w.astype(np.float32) * a_w[..., None]
    out = out * (1.0 - a_stat[..., None]) + plate.astype(np.float32) * a_stat[..., None]
    # lock the window pane (minus a thin fringe of cat) to the unwarped plate
    pane = pane_f()
    cat_close = cv2.dilate((a_c > 0.15).astype(np.uint8),
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    lock = (pane > 0.5) & (cat_close == 0) & (a_stat < 0.05)
    out[lock] = plate[lock].astype(np.float32)
    off = (a_w < 0.02) & (a_stat < 0.02) & (dil < 0.02)
    out[off] = plate[off].astype(np.float32)
    y0, y1, x0, x1 = TITLE
    out[y0:y1, x0:x1] = plate[y0:y1, x0:x1]
    return np.clip(out, 0, 255).astype(np.uint8)



# -------------- chin hinge --------------
HINGE_Y = 388.0
CHIN_REST = 407.0
JAW_KEEP = 0.60          # keep 60% of the current lower-jaw drop
NECK_Y1 = 530.0


def lower_lip_y(im):
    """Bottom of the mouth cavity / tongue in a tight muzzle box."""
    y0, y1, x0, x1 = 370, 455, 680, 780
    sl = im[y0:y1, x0:x1]
    imf = sl.astype(np.float32)
    B, G, R = imf[..., 0], imf[..., 1], imf[..., 2]
    Yl = 0.114 * B + 0.587 * G + 0.299 * R
    cav = (Yl < 68) & (Yl > 5)
    tongue = (R > G + 12) & (R > B + 8) & (Yl < 165) & (Yl > 30)
    m = (cav | tongue).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    # rows with enough cavity pixels
    rs = m.sum(axis=1)
    hits = np.where(rs >= 6)[0]
    if hits.size < 2:
        return None
    return float(y0 + hits[-1])


def jaw_strength(local, use2):
    """1 at the meow peak, 0 outside the window."""
    peak = 191 if use2 else 138
    pre, post = 18, 22
    if local < peak - pre or local > peak + post:
        return 0.0
    if local <= peak:
        return float(smootherstep((local - (peak - pre)) / max(pre, 1)))
    return float(1.0 - smootherstep((local - peak) / max(post, 1)))


def apply_chin_hinge(im, local, use2, a_cat):
    """Squash the lower jaw toward a hinge; slide the neck without scaling."""
    k = jaw_strength(local, use2)
    if k < 0.05:
        return im
    ly = lower_lip_y(im)
    if ly is None:
        return im
    drop = ly - CHIN_REST
    if drop < 3.0:
        return im
    # we want the lip to land at CHIN_REST + JAW_KEEP*drop
    new_lip = CHIN_REST + JAW_KEEP * drop
    # inverse map for dest y
    y0, y1 = 360, int(NECK_Y1) + 8
    x0, x1 = 600, 880
    bh, bw = y1 - y0, x1 - x0
    yy, xx = np.mgrid[0:bh, 0:bw]
    dest_y = yy + y0
    hinge = HINGE_Y
    src_y = dest_y.astype(np.float32)
    # between hinge and new_lip: expand back to [hinge, old lip]
    mid = (dest_y > hinge) & (dest_y <= new_lip)
    src_y = np.where(mid, hinge + (dest_y - hinge) * (drop / max(JAW_KEEP * drop, 1e-3)), src_y)
    # below new_lip: neck 1:1 from old lip downward (slide up, no scale)
    below = dest_y > new_lip
    src_y = np.where(below, ly + (dest_y - new_lip), src_y)
    map_x = (xx + x0).astype(np.float32)
    map_y = np.clip(src_y, 0, H - 1).astype(np.float32)
    warped = cv2.remap(im, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    # spatial weight: muzzle column, fade at sides and below chest
    cx = 725.0
    wx = np.exp(-0.5 * ((xx + x0 - cx) / 78.0) ** 2)
    wy = np.clip((dest_y - (hinge - 4.0)) / 8.0, 0, 1) * np.clip((NECK_Y1 - dest_y) / 18.0, 0, 1)
    a = (a_cat[y0:y1, x0:x1] if a_cat is not None else 1.0)
    if not np.isscalar(a):
        a = np.clip(a, 0, 1)
        a = cv2.GaussianBlur(a.astype(np.float32), (0, 0), 3.0)
    w = (wx * wy * a * k).astype(np.float32)
    # decay to 0 before the rectangle cut so the hinge box cannot print
    # a straight card edge on the cheek or chest
    edge = 28.0
    fx = np.clip(xx / edge, 0, 1) * np.clip((bw - 1 - xx) / edge, 0, 1)
    fy = np.clip(yy / edge, 0, 1) * np.clip((bh - 1 - yy) / edge, 0, 1)
    fx = fx * fx * (3.0 - 2.0 * fx)
    fy = fy * fy * (3.0 - 2.0 * fy)
    w = w * fx * fy
    w = cv2.GaussianBlur(w, (0, 0), 1.6)
    sl = im[y0:y1, x0:x1].astype(np.float32)
    out = sl * (1.0 - w[..., None]) + warped.astype(np.float32) * w[..., None]
    im = im.copy()
    im[y0:y1, x0:x1] = np.clip(out, 0, 255).astype(np.uint8)
    return im


# -------------- I2V blink composite --------------
# Source clip lids are natural at 3× when closed (i2v 32–36). Alignment is
# computed on OPEN eyes (I2V-24 vs the current still-open target) and then
# applied to the closing I2V frames so closed lids are never used as the
# flow target. Fast close, short hold, slower open. 2 blinks / cycle, on
# settle-beat holds, away from seams, meows, and ear twitches.
I2V_OPEN = 24
# t=0..12  (close 4, hold 2, open 7) -> I2V source frames
I2V_SEQ = (26, 28, 30, 32, 34, 36, 37, 38, 40, 41, 42, 43, 44)
BLINK_CLOSE, BLINK_HOLD, BLINK_OPEN = 4, 2, 7
BLINK_LEN = BLINK_CLOSE + BLINK_HOLD + BLINK_OPEN  # 13
BLINK_STARTS = (40, 154)  # local; hold A and hold C
_I2V_FRAMES = None
_I2V_AFFINE_TO_N0 = None


def load_i2v_blinks():
    global _I2V_FRAMES, _I2V_AFFINE_TO_N0
    if _I2V_FRAMES is not None:
        return _I2V_FRAMES
    if not BLINK_I2V.exists():
        print('I2V blink clip missing', BLINK_I2V, flush=True)
        _I2V_FRAMES = []
        return _I2V_FRAMES
    cap = cv2.VideoCapture(str(BLINK_I2V))
    fr = []
    while True:
        ok, im = cap.read()
        if not ok:
            break
        fr.append(im)
    cap.release()
    _I2V_FRAMES = fr
    print(f'loaded I2V blink clip {len(fr)} frames', flush=True)
    return fr


def _face_flow_affine(a, b, mask):
    """Partial affine mapping a -> b from DIS flow in mask."""
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    f = dis.calc(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY),
                 cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), None)
    ys, xs = np.where(mask > 0)
    if xs.size < 80:
        return np.array([[1, 0, 0], [0, 1, 0]], np.float32)
    step = max(1, xs.size // 800)
    xs, ys = xs[::step], ys[::step]
    src = np.stack([xs, ys], 1).astype(np.float32)
    dst = src + f[ys, xs]
    M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    if M is None:
        return np.array([[1, 0, 0], [0, 1, 0]], np.float32)
    return M.astype(np.float32)


def _eye_mask_rest():
    m = np.zeros((H, W), np.float32)
    for (cx, cy, rx, ry) in ((665, 310, 38, 28), (780, 322, 42, 30)):
        cv2.ellipse(m, (int(cx), int(cy)), (int(rx), int(ry)), 0, 0, 360, 1.0, -1)
    m = cv2.GaussianBlur(m, (0, 0), 6.0)
    return np.clip(m, 0, 1)


def blink_t(local):
    """Offset into the current blink, or None."""
    for s in BLINK_STARTS:
        t = int(local) - int(s)
        if 0 <= t < BLINK_LEN:
            return t
    return None


def i2v_src_index(local):
    t = blink_t(local)
    if t is None:
        return None
    return int(I2V_SEQ[min(t, len(I2V_SEQ) - 1)])


def blink_close_k(local):
    """0 open .. 1 fully closed."""
    t = blink_t(local)
    if t is None:
        return 0.0
    if t < BLINK_CLOSE:
        return float(smootherstep((t + 0.5) / BLINK_CLOSE))
    if t < BLINK_CLOSE + BLINK_HOLD:
        return 1.0
    return float(1.0 - smootherstep((t - BLINK_CLOSE - BLINK_HOLD + 0.5) / BLINK_OPEN))


def _eye_affine(src, dst, cx, cy, rx=54, ry=46):
    """Open-eye affine src -> dst in one eye neighbourhood."""
    m = np.zeros((H, W), np.uint8)
    cv2.ellipse(m, (int(cx), int(cy)), (int(rx), int(ry)), 0, 0, 360, 1, -1)
    return _face_flow_affine(src, dst, m)


def _compose_affine(A, B):
    """2x3 affines: apply A first, then B."""
    A3 = np.vstack([A, np.array([0.0, 0.0, 1.0], np.float32)])
    B3 = np.vstack([B, np.array([0.0, 0.0, 1.0], np.float32)])
    C = B3 @ A3
    return C[:2].astype(np.float32)


def _lid_mask_i2v(fr, si, k):
    """Opaque lid / changed-eye mask in I2V space, k=close amount."""
    open_im = fr[I2V_OPEN]
    cur = fr[si]
    d = np.abs(luma(cur) - luma(open_im))
    changed = (d > 6.0).astype(np.uint8)
    m = np.zeros((H, W), np.uint8)
    # slightly squashed ellipses that grow with close
    for (cx, cy, rx, ry) in ((665, 309, 40, 26), (780, 322, 44, 28)):
        ryy = max(8, int(round(ry * (0.35 + 0.80 * k))))
        cv2.ellipse(m, (int(cx), int(cy)), (int(rx), ryy), 0, 0, 360, 1, -1)
    # keep changes only near the eyes
    near = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
    m = np.maximum(m, changed * near)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    # hard cover, then a short feather
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    mf = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 2.2)
    # keep the dilated interior at 1 so irises cannot leak
    core = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    mf = np.maximum(mf, core.astype(np.float32))
    return np.clip(mf, 0, 1)


def apply_i2v_blink(im, local, ref0=None):
    """Paste I2V lids onto the current open-eyed face, open-eye aligned."""
    si = i2v_src_index(local)
    if si is None:
        return im
    k = blink_close_k(local)
    if k < 0.04:
        return im
    fr = load_i2v_blinks()
    if si >= len(fr) or len(fr) <= I2V_OPEN:
        return im
    i2v_open = fr[I2V_OPEN]
    i2v_k = fr[si]
    # per-eye affine I2V-open -> current target (both still have open eyes)
    ML = _eye_affine(i2v_open, im, 665, 309)
    MR = _eye_affine(i2v_open, im, 780, 322)
    # warp each I2V eye with its own affine, then combine
    srcL = cv2.warpAffine(i2v_k, ML, (W, H), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)
    srcR = cv2.warpAffine(i2v_k, MR, (W, H), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)
    mid = 0.5 * (665.0 + 780.0)
    # soft seam down the bridge of the nose
    blend = np.clip((Xg - (mid - 18.0)) / 36.0, 0, 1).astype(np.float32)
    src = srcL.astype(np.float32) * (1.0 - blend[..., None]) + srcR.astype(np.float32) * blend[..., None]
    maskL = cv2.warpAffine(_lid_mask_i2v(fr, si, k), ML, (W, H),
                           flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    maskR = cv2.warpAffine(_lid_mask_i2v(fr, si, k), MR, (W, H),
                           flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    em = np.clip(np.maximum(maskL, maskR), 0, 1)
    # force-cover the target's current iris so it cannot ghost under the lid
    Yl = luma(im)
    iris = np.zeros((H, W), np.uint8)
    for (cx, cy, rx, ry) in ((665, 309, 36, 24), (780, 322, 40, 26)):
        cv2.ellipse(iris, (int(cx), int(cy)), (int(rx), int(ry)), 0, 0, 360, 1, -1)
    iris = (iris > 0) & (Yl < 85)
    iris = cv2.dilate(iris.astype(np.uint8),
                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    # once the lids are mid-close or further, cover the target iris fully;
    # scaling this by k is what previously ghosted an open eye under the lid
    iris_w = 1.0 if k >= 0.28 else float(k / 0.28)
    em = np.clip(np.maximum(em, iris.astype(np.float32) * iris_w), 0, 1)
    # luma-match the lid patch to the surrounding face
    ring = cv2.dilate((em > 0.3).astype(np.uint8),
                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    ring = (ring > 0) & (em < 0.2)
    sl = src
    if int(ring.sum()) > 40:
        src_l = float(luma(np.clip(sl, 0, 255).astype(np.uint8))[ring].mean())
        dst_l = float(Yl[ring].mean())
        sc = np.clip(dst_l / max(src_l, 1.0), 0.88, 1.12)
        sl = np.clip(sl * sc, 0, 255)
    # The I2V frame already carries the close amount. Keep the lid/iris core
    # opaque so mid-close cannot blend the open eye back through. Ease only
    # the first and last tick of the cycle.
    t = blink_t(local)
    w = em
    if t == 0 or t == BLINK_LEN - 1:
        w = em * 0.70
    w = np.clip(w, 0, 1)[..., None]
    out = im.astype(np.float32) * (1.0 - w) + sl * w
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_mouth_hook(out, i, **kwargs):
    ref = MOUTH_REF_PATH
    if not ref:
        return out
    path = Path(ref)
    if not path.exists():
        print(f'warning: mouth ref missing {path}', flush=True)
        return out
    return out  # swap-in when the path arrives


# -------------- process --------------


# g1v3/g2v3 cat-solid has a baked-in rectangular hole over the face
# (x=610..828, y=0..390) on every frame. That punched u-ex through the
# gen face and printed the card edge at x=829 / y=391. Repair in memory
# only; do not rewrite the npz mattes.
_CAT_HOLE = (0, 391, 610, 829)  # y0,y1,x0,x1 exclusive-end


def repair_cat_matte(a, gen_im):
    """Fill the static face-card hole using fur in the hole box only."""
    a = a.astype(np.float32)
    if a.max() > 1.5:
        a = a / 255.0
    Yl = luma(gen_im)
    B = gen_im[..., 0].astype(np.float32)
    G = gen_im[..., 1].astype(np.float32)
    R = gen_im[..., 2].astype(np.float32)
    cream = cream_mask(gen_im)
    orange = (R > G + 3) & (R > B - 2) & (Yl > 26) & (Yl < 235)
    dark_fur = ((Yl > 18) & (Yl < 95) & (R + 20 >= G)
                & ~((B > R + 8) & (B > G + 8)))
    fur = (cream | orange | dark_fur).astype(np.uint8)
    sky = (Yl < 50) & (B > R + 4) & (B > G)
    y0, y1, x0, x1 = _CAT_HOLE
    fill = np.zeros(a.shape, np.uint8)
    fill[y0:y1, x0:x1] = fur[y0:y1, x0:x1] & (~sky[y0:y1, x0:x1])
    fill = cv2.morphologyEx(fill, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(fill, 8)
    keep = np.zeros_like(fill)
    for i in range(1, n):
        bot = int(st[i, cv2.CC_STAT_TOP] + st[i, cv2.CC_STAT_HEIGHT] - 1)
        if st[i, cv2.CC_STAT_AREA] >= 80 and bot >= 200:
            keep[lab == i] = 1
    solid = np.zeros_like(keep)
    cnts, _ = cv2.findContours(keep, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        cv2.drawContours(solid, cnts, -1, 1, thickness=-1)
    # shave thin sky spikes above a smoothed crown envelope
    tops = np.full(x1 - x0, -1, np.int32)
    for i, x in enumerate(range(x0, x1)):
        ys = np.where(solid[:y1, x] > 0)[0]
        if ys.size:
            tops[i] = int(ys.min())
    valid = tops >= 0
    if valid.sum() >= 8:
        t = tops.astype(np.float32)
        t[~valid] = np.interp(np.where(~valid)[0], np.where(valid)[0], t[valid])
        k = np.ones(31, np.float32) / 31.0
        sm = np.convolve(np.pad(t, 15, mode='edge'), k, mode='valid')
        for i, x in enumerate(range(x0, x1)):
            cut = int(round(sm[i])) - 2
            if tops[i] >= 0 and tops[i] < cut - 4:
                solid[tops[i]:cut, x] = 0
    fa = cv2.GaussianBlur(solid.astype(np.float32), (0, 0), 1.8)
    out = np.clip(np.maximum(a, fa * 0.992), 0, 1)
    # close ear-wedge gaps: horizontal bounded fill in the head band
    yb0, yb1, xb0, xb1 = 150, 420, 540, 920
    sl = out[yb0:yb1, xb0:xb1]
    firm = sl > 0.5
    h, w = sl.shape
    for y in range(h):
        row = firm[y]
        x = 0
        while x < w:
            if row[x]:
                x += 1
                continue
            xa = x
            while x < w and not row[x]:
                x += 1
            xb = x
            if xa > 0 and xb < w and (xb - xa) <= 280:
                fillv = max(float(sl[y, xa - 1]), float(sl[y, min(xb, w - 1)]))
                sl[y, xa:xb] = fillv
    out[yb0:yb1, xb0:xb1] = sl
    return np.clip(out, 0, 1)


def repair_cat_stack(stack, gen_stack):
    """In-memory repair of a uint8 cat-solid stack. Does not write the npz."""
    out = np.empty_like(stack)
    for i in range(len(stack)):
        r = repair_cat_matte(stack[i], gen_stack[i])
        out[i] = np.clip(np.round(r * 255.0), 0, 255).astype(np.uint8)
    return out


# Face-card rectangle observed on encoded ad-ex meows: right edge x=830
# (= MOUTH_BOX x1) and a horizontal step near y=455, extending up through
# the forehead (~y=200). Not in the gen source.
FACE_CARD = (200, 470, 620, 830)  # y0,y1,x0,x1


def _ss(t):
    t = np.clip(t, 0, 1)
    return t * t * (3.0 - 2.0 * t)


def heal_box_edges(im, box, width=22):
    """Match low-frequency brightness across a rectangular paste edge.

    Fur texture (high frequency) is kept. Nearly a no-op when both sides
    of the edge already agree.
    """
    y0, y1, x0, x1 = box
    out = im.astype(np.float32).copy()
    low = cv2.GaussianBlur(out, (0, 0), 5.0)
    high = out - low

    def _vert(xedge):
        nonlocal out
        xa = max(0, int(xedge) - width)
        xb = min(W, int(xedge) + width)
        if xb - xa < 6:
            return
        ya = max(0, int(y0) - width)
        yb = min(H, int(y1) + width)
        left = low[ya:yb, xa:xa + 4].mean(axis=1, keepdims=True)
        right = low[ya:yb, xb - 4:xb].mean(axis=1, keepdims=True)
        n = xb - xa
        t = _ss(np.linspace(0, 1, n, dtype=np.float32))[None, :, None]
        interp = left * (1.0 - t) + right * t
        wr = np.hanning(n).astype(np.float32)[None, :, None]
        vy = np.ones((yb - ya, 1, 1), np.float32)
        fade = min(width, 16)
        if fade > 1:
            ramp = _ss(np.linspace(0, 1, fade, dtype=np.float32))
            vy[:fade, 0, 0] = ramp
            vy[-fade:, 0, 0] = ramp[::-1]
        m = wr * vy
        patch = low[ya:yb, xa:xb] * (1.0 - m) + interp * m
        out[ya:yb, xa:xb] = np.clip(patch + high[ya:yb, xa:xb], 0, 255)

    def _horiz(yedge):
        nonlocal out
        ya = max(0, int(yedge) - width)
        yb = min(H, int(yedge) + width)
        if yb - ya < 6:
            return
        xa = max(0, int(x0) - width)
        xb = min(W, int(x1) + width)
        top = low[ya:ya + 4, xa:xb].mean(axis=0, keepdims=True)
        bot = low[yb - 4:yb, xa:xb].mean(axis=0, keepdims=True)
        n = yb - ya
        t = _ss(np.linspace(0, 1, n, dtype=np.float32))[:, None, None]
        interp = top * (1.0 - t) + bot * t
        wr = np.hanning(n).astype(np.float32)[:, None, None]
        vx = np.ones((1, xb - xa, 1), np.float32)
        fade = min(width, 16)
        if fade > 1:
            ramp = _ss(np.linspace(0, 1, fade, dtype=np.float32))
            vx[0, :fade, 0] = ramp
            vx[0, -fade:, 0] = ramp[::-1]
        m = wr * vx
        patch = low[ya:yb, xa:xb] * (1.0 - m) + interp * m
        out[ya:yb, xa:xb] = np.clip(patch + high[ya:yb, xa:xb], 0, 255)

    _vert(x0)
    _vert(x1)
    _horiz(y0)
    _horiz(y1)
    return np.clip(out, 0, 255).astype(np.uint8)



def soften_hairlines(im, xs=(), ys=(), width=11, yband=(180, 480), xband=(580, 900)):
    """Full-spectrum blend across a known 1 px rectangular border, face only."""
    out = im.astype(np.float32)
    h, w = out.shape[:2]
    y0, y1 = yband
    x0, x1 = xband
    for xedge in xs:
        xa = max(0, int(xedge) - width)
        xb = min(w, int(xedge) + width)
        if xb - xa < 4:
            continue
        sl = out[y0:y1, xa:xb]
        left = out[y0:y1, xa:xa + 3].mean(axis=1, keepdims=True)
        right = out[y0:y1, xb - 3:xb].mean(axis=1, keepdims=True)
        n = xb - xa
        t = _ss(np.linspace(0, 1, n, dtype=np.float32))[None, :, None]
        wr = np.hanning(n).astype(np.float32)[None, :, None]
        # fade the blend at the top/bottom of the face band
        fade = np.ones((y1 - y0, 1, 1), np.float32)
        f = 16
        fade[:f, 0, 0] = _ss(np.linspace(0, 1, f, dtype=np.float32))
        fade[-f:, 0, 0] = _ss(np.linspace(1, 0, f, dtype=np.float32))
        m = wr * fade
        interp = left * (1.0 - t) + right * t
        out[y0:y1, xa:xb] = sl * (1.0 - m) + interp * m
    for yedge in ys:
        ya = max(0, int(yedge) - width)
        yb = min(h, int(yedge) + width)
        if yb - ya < 4:
            continue
        sl = out[ya:yb, x0:x1]
        top = out[ya:ya + 3, x0:x1].mean(axis=0, keepdims=True)
        bot = out[yb - 3:yb, x0:x1].mean(axis=0, keepdims=True)
        n = yb - ya
        t = _ss(np.linspace(0, 1, n, dtype=np.float32))[:, None, None]
        wr = np.hanning(n).astype(np.float32)[:, None, None]
        xx = np.arange(x0, x1, dtype=np.float32)
        # skip the mouth opening
        muzzle = np.clip((np.abs(xx - 725.0) - 55.0) / 18.0, 0, 1)
        muzzle = (muzzle * muzzle * (3.0 - 2.0 * muzzle))[None, :, None]
        # fade at the sides of the x band
        side = np.ones((1, x1 - x0, 1), np.float32)
        f = 16
        side[0, :f, 0] = _ss(np.linspace(0, 1, f, dtype=np.float32))
        side[0, -f:, 0] = _ss(np.linspace(1, 0, f, dtype=np.float32))
        m = wr * muzzle * side
        interp = top * (1.0 - t) + bot * t
        out[ya:yb, x0:x1] = sl * (1.0 - m) + interp * m
    return np.clip(out, 0, 255).astype(np.uint8)



def hide_small_tail_gaps(im, uex, a_p):
    """Cover small cream fragments that peek through lower leaf gaps.

    The main tail mass (large component on the left of the plant) is kept.
    Isolated cream blobs under 800 px in the lower plant are replaced with
    the u-ex plate (usually leaf/shadow in those gaps).
    """
    cream = cream_mask(im)
    roi = (Xg > 970) & (Yg > 500)
    m = (cream & roi).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    hide = np.zeros(m.shape, np.uint8)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < 800:
            hide[lab == i] = 1
    if int(hide.sum()) < 4:
        return im
    hide = cv2.dilate(hide, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    # also prefer leaf pixels: boost where the plant matte is already on
    a = np.clip(a_p, 0, 1) if a_p is not None else 0
    h = hide.astype(np.float32)
    if not np.isscalar(a):
        h = np.clip(h + 0.35 * a * hide, 0, 1)
    h = cv2.GaussianBlur(h, (0, 0), 1.3)
    # if the plate itself is cream here, inpaint from nearby non-cream plate
    plate = uex.astype(np.float32)
    cream_u = cream_mask(uex)
    bad = (h > 0.3) & cream_u
    if int(bad.sum()) > 4:
        inpaint = cv2.inpaint(uex, bad.astype(np.uint8), 5, cv2.INPAINT_TELEA).astype(np.float32)
        plate = plate * (1.0 - bad[..., None].astype(np.float32)) + inpaint * bad[..., None].astype(np.float32)
    out = im.astype(np.float32) * (1.0 - h[..., None]) + plate * h[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def process(src_u, c1, c2, p1, p2, h1, h2, gi1, gi2, w_room, w_bulb, preview=None):
    want = set(preview) if preview else set(range(NFR))
    out = [None] * NFR
    global _eye_priors
    _eye_priors = dict(EYE_PRIOR)
    start1 = (gi1[0].astype(np.float32), c1[0]/255.0, p1[0]/255.0, h1[0]/255.0)
    start2 = (gi2[0].astype(np.float32), c2[0]/255.0, p2[0]/255.0, h2[0]/255.0)
    ph0 = Hh.phase_of(0)
    rest_pose = [None, None, None]
    for i in range(NFR):
        if i not in want:
            continue
        use2, local = Y.cycle_of(i)
        gen, a_c, a_p, a_h = sample_layers(use2, local, None, None, c1, c2, p1, p2, h1, h2, gi1, gi2)
        w = Z.in_seam(i)
        if w is not None:
            tgt = start2 if Z.next_start_is_gen2(i) else start1
            gen0 = gen
            gen, a_c = Z.flow_blend(gen0, tgt[0], a_c, tgt[1], w)
            _, a_p = Z.flow_blend(gen0, tgt[0], a_p, tgt[2], w)
            _, a_h = Z.flow_blend(gen0, tgt[0], a_h, tgt[3], w)
            gen = np.clip(gen, 0, 255).astype(np.uint8)
        ph = Hh.phase_of(i)
        a_pu = u_plant_matte(src_u)
        plate, a_pu = compose_layered(src_u[i], gen, a_c, a_p, a_h, ph, ph0, w_room, w_bulb, a_pu)
        dx, dy = warp_field(i)
        # confine the mesh strictly to the cat (tighter than ac-ex)
        gate = cv2.dilate((a_c > 0.08).astype(np.uint8),
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        gate = cv2.GaussianBlur(gate.astype(np.float32), (0, 0), 2.0)
        # never warp the window pane
        pane = pane_f()
        gate = gate * (1.0 - np.clip(pane * (1.0 - cv2.dilate((a_c > 0.2).astype(np.uint8),
                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))), 0, 1))
        dx, dy = dx * gate, dy * gate
        im = compose_warped(plate, src_u[i], a_c, a_pu, a_h, dx, dy)
        im = hide_small_tail_gaps(im, src_u[i], a_pu)
        im = apply_chin_hinge(im, local, use2, a_c)
        # residual 1 px hairlines can remain at the repaired hole borders
        im = soften_hairlines(im, xs=(610, 829), ys=(391,), width=11)
        cyc = 0 if i < 241 else (1 if i < 482 else 2)
        if rest_pose[cyc] is None and local <= 2:
            rest_pose[cyc] = im.copy()
        im = apply_i2v_blink(im, local, rest_pose[cyc] if rest_pose[cyc] is not None else im)
        im = apply_mouth_hook(im, i)
        out[i] = im
        if preview or i % 25 == 0 or w is not None or (use2 and 188 <= local <= 204) or i2v_src_index(local) is not None:
            extra = f' seam_w={w:.3f}' if w is not None else ''
            bi = i2v_src_index(local)
            btag = f' i2v={bi}' if bi is not None else ''
            print(f'  n{i} local={local} k={K_BREATH[local]:.2f} '
                  f'dy_chest={-CHEST_AMP*K_BREATH[local]:.2f}{btag}{extra}',
                  flush=True)
    return out


# -------------- QC --------------
def seam_region_report(out, dest, src_u=None):
    dest.mkdir(parents=True, exist_ok=True)
    typ = list(range(10, 30)) + list(range(50, 70)) + list(range(200, 220))
    boxes = {
        'head': HEAD,
        'steam': STEAM_BOX,
        'tail': TAIL_BOX,
    }
    lines = ['pair  region  mad_rgb  mad_luma']
    typ_vals = {k: [] for k in boxes}
    for i in typ:
        if out[i] is None or out[i + 1] is None:
            continue
        for k, box in boxes.items():
            typ_vals[k].append(mad_rgb(out[i], out[i + 1], box))
    for k in boxes:
        if typ_vals[k]:
            lines.append(f'typical  {k}  {float(np.mean(typ_vals[k])):.4f}')
    pairs = [(239, 240), (240, 241), (241, 242),
             (480, 481), (481, 482), (482, 483), (0, 1), (10, 11)]
    for a, b in pairs:
        if out[a] is None or out[b] is None:
            continue
        for k, box in boxes.items():
            r, l = mad_rgb(out[a], out[b], box), mad_luma(out[a], out[b], box)
            lines.append(f'{a}-{b}  {k}  {r:.4f}  {l:.4f}')
    path = dest / 'seam-mad-regions.txt'
    path.write_text('\n'.join(lines) + '\n')
    print('SEAM REGIONS\n' + '\n'.join(lines), flush=True)

    def crop(im, box, sc, fy0=None):
        y0, y1, x0, x1 = box
        return cv2.resize(im[y0:y1, x0:x1], None, fx=sc, fy=sc,
                          interpolation=cv2.INTER_NEAREST)

    for a, b, name in ((240, 241, '240-241'), (481, 482, '481-482')):
        if out[a] is None or out[b] is None:
            continue
        for tag, box, sc in (('steam', STEAM_BOX, 2), ('tail', TAIL_BOX, 2),
                             ('head', (140, 280, 560, 940), 2)):
            row = np.hstack([crop(out[a], box, sc), crop(out[b], box, sc)])
            cv2.putText(row, f'{tag} {a}', (8, 24), cv2.FONT_HERSHEY_PLAIN, 1.6, (255, 255, 255), 2)
            cv2.putText(row, f'{b}', (row.shape[1] // 2 + 8, 24),
                        cv2.FONT_HERSHEY_PLAIN, 1.6, (255, 255, 255), 2)
            cv2.imwrite(str(dest / f'seam-{tag}-{name}.png'), row)
            d = np.abs(out[a].astype(np.int16) - out[b].astype(np.int16))
            y0, y1, x0, x1 = box
            vis = cv2.resize(np.clip(d[y0:y1, x0:x1] * 8, 0, 255).astype(np.uint8),
                             None, fx=sc, fy=sc, interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(str(dest / f'seam-{tag}-diff-{name}-x8.png'), vis)
    return path


def mouth_seq(out, dest):
    dest.mkdir(parents=True, exist_ok=True)
    cells = []
    for i in range(420, 446):
        if out[i] is None:
            continue
        c = cv2.resize(out[i][200:430, 560:880], None, fx=3, fy=3,
                       interpolation=cv2.INTER_NEAREST)
        cv2.putText(c, f'n{i}', (8, 28), cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255), 2)
        cells.append(c)
        cv2.imwrite(str(dest / f'mouth-n{i}.png'), c)
    if cells:
        mid = max(1, (len(cells) + 1) // 2)
        r1 = np.hstack(cells[:mid])
        if cells[mid:]:
            r2 = np.hstack(cells[mid:])
            if r2.shape[1] < r1.shape[1]:
                r2 = np.hstack([r2, np.zeros((r2.shape[0], r1.shape[1] - r2.shape[1], 3), np.uint8)])
            sheet = np.vstack([r1, r2])
        else:
            sheet = r1
        cv2.imwrite(str(dest / 'meow-seq-432-every.png'), sheet)


def tail_closeups(out, dest):
    dest.mkdir(parents=True, exist_ok=True)
    cells = []
    for i in (0, 60, 72, 78, 84, 90, 96, 102, 108, 114, 120, 180, 240, 241, 313, 349, 432, 481, 482):
        if i >= len(out) or out[i] is None:
            continue
        c = cv2.resize(out[i][400:720, 880:1280], None, fx=2, fy=2,
                       interpolation=cv2.INTER_NEAREST)
        cv2.putText(c, f'n{i}', (8, 24), cv2.FONT_HERSHEY_PLAIN, 1.6, (255, 255, 255), 2)
        cells.append(c)
        cv2.imwrite(str(dest / f'tail-n{i}.png'), c)
    if cells:
        # rows of 6
        rows = []
        for j in range(0, len(cells), 6):
            chunk = cells[j:j + 6]
            if len(chunk) < 6:
                pad = [np.zeros_like(cells[0]) for _ in range(6 - len(chunk))]
                chunk = chunk + pad
            rows.append(np.hstack(chunk))
        cv2.imwrite(str(dest / 'tail-closeup-grid.png'), np.vstack(rows))


def blink_frame_ranges():
    # every blink frame, plus one open frame on each side
    out = []
    for cyc, off in enumerate((0, 241, 482)):
        for j, s in enumerate(BLINK_STARTS):
            a = off + s - 1
            b = off + s + BLINK_LEN + 1
            a = max(0, min(NFR - 1, a))
            b = max(0, min(NFR, b))
            out.append((f'{chr(65+j)}{cyc}', list(range(a, b))))
    return out


def _eye_crop3(im, box):
    y0, y1, x0, x1 = box
    return cv2.resize(im[y0:y1, x0:x1], None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)


def blink_strips(out, dest):
    dest.mkdir(parents=True, exist_ok=True)
    face = (230, 370, 620, 920)   # user crop
    eyeL = (250, 360, 620, 740)
    eyeR = (250, 360, 720, 860)
    all_face, all_L, all_R = [], [], []
    for tag, idxs in blink_frame_ranges():
        faces, Ls, Rs = [], [], []
        for i in idxs:
            if i >= len(out) or out[i] is None:
                continue
            f = _eye_crop3(out[i], face)
            L = _eye_crop3(out[i], eyeL)
            R = _eye_crop3(out[i], eyeR)
            for im, lab in ((f, f'n{i}'), (L, f'L n{i}'), (R, f'R n{i}')):
                cv2.putText(im, lab, (8, 24), cv2.FONT_HERSHEY_PLAIN, 1.6, (255, 255, 255), 2)
            faces.append(f); Ls.append(L); Rs.append(R)
            all_face.append(f); all_L.append(L); all_R.append(R)
            cv2.imwrite(str(dest / f'blink-{tag}-n{i:03d}-face3x.png'), f)
            cv2.imwrite(str(dest / f'blink-{tag}-n{i:03d}-L3x.png'), L)
            cv2.imwrite(str(dest / f'blink-{tag}-n{i:03d}-R3x.png'), R)
        if faces:
            cv2.imwrite(str(dest / f'blink-{tag}-face-strip.png'), np.hstack(faces))
            cv2.imwrite(str(dest / f'blink-{tag}-L-strip.png'), np.hstack(Ls))
            cv2.imwrite(str(dest / f'blink-{tag}-R-strip.png'), np.hstack(Rs))
    if all_face:
        # one contact sheet of every blink frame, both eyes (face crop)
        # wrap into two rows if very wide
        row = np.hstack(all_face)
        cv2.imwrite(str(dest / 'blink-ALL-face-strip.png'), row)
        cv2.imwrite(str(dest / 'blink-ALL-L-strip.png'), np.hstack(all_L))
        cv2.imwrite(str(dest / 'blink-ALL-R-strip.png'), np.hstack(all_R))


def motion_strips(out, dest):
    dest.mkdir(parents=True, exist_ok=True)
    # breath chest
    cells = []
    for i in list(range(16, 50, 2)):
        if out[i] is None:
            continue
        c = out[i][400:620, 540:900].copy()
        cv2.putText(c, f'n{i}', (6, 20), cv2.FONT_HERSHEY_PLAIN, 1.3, (255, 255, 255), 1)
        cells.append(c)
    if cells:
        cv2.imwrite(str(dest / 'breath-chest-strip.png'), np.hstack(cells))
    # face drift over a rest stretch
    cells = []
    for i in list(range(16, 50, 2)) + list(range(90, 120, 2)):
        if i >= len(out) or out[i] is None:
            continue
        c = out[i][180:430, 520:900].copy()
        cv2.putText(c, f'n{i}', (6, 20), cv2.FONT_HERSHEY_PLAIN, 1.3, (255, 255, 255), 1)
        cells.append(c)
    if cells:
        cv2.imwrite(str(dest / 'face-rest-strip.png'), np.hstack(cells))
    for start, tag, box in ((70, 'L', (140, 280, 520, 760)),
                            (347, 'R', (140, 280, 700, 960))):
        cells = []
        for i in range(start, start + 14):
            if i >= len(out) or out[i] is None:
                continue
            a = cv2.resize(out[i][box[0]:box[1], box[2]:box[3]], None,
                           fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
            cv2.putText(a, f'n{i}', (6, 22), cv2.FONT_HERSHEY_PLAIN, 1.4, (255, 255, 255), 1)
            cells.append(a)
        if cells:
            cv2.imwrite(str(dest / f'twitch-{tag}-strip.png'), np.hstack(cells))
    # head turn / settle (parallax check)
    cells = []
    for i in list(range(26, 36)) + list(range(50, 62)) + list(range(66, 78)) + list(range(110, 122)):
        if i >= len(out) or out[i] is None:
            continue
        c = cv2.resize(out[i][160:430, 540:920], None, fx=2, fy=2,
                       interpolation=cv2.INTER_NEAREST)
        cv2.putText(c, f'n{i}', (6, 22), cv2.FONT_HERSHEY_PLAIN, 1.6, (255, 255, 255), 2)
        cells.append(c)
        cv2.imwrite(str(dest / f'head-turn-n{i:03d}.png'), c)
    if cells:
        # wrap in rows of 8
        cols = 8
        rows = []
        for r in range(0, len(cells), cols):
            chunk = cells[r:r + cols]
            if len(chunk) < cols:
                chunk = chunk + [np.zeros_like(cells[0])] * (cols - len(chunk))
            rows.append(np.hstack(chunk))
        cv2.imwrite(str(dest / 'head-turn-strip.png'), np.vstack(rows))


def edge_qc(out, src_u, dest):
    dest.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(0, 600, 10):
        if out[i] is None:
            continue
        d = cv2.absdiff(out[i], src_u[i]).astype(np.float32)
        y0, y1, x0, x1 = TITLE
        title = float(d[y0:y1, x0:x1].mean())
        desk = float(d[400:720, 80:500].mean())
        frame = float(d[40:160, 250:360].mean())
        steam = float(d[STEAM_BOX[0]:STEAM_BOX[1], STEAM_BOX[2]:STEAM_BOX[3]].mean())
        rows.append((i, title, desk, frame, steam, float(d.mean())))
    path = dest / 'edge-mae.txt'
    with open(path, 'w') as f:
        f.write('i title desk left_frame steam global\n')
        for r in rows:
            f.write(f'{r[0]:4d}  {r[1]:.4f}  {r[2]:.4f}  {r[3]:.4f}  {r[4]:.4f}  {r[5]:.4f}\n')
    print('edge mae title', np.mean([r[1] for r in rows]),
          'desk', np.mean([r[2] for r in rows]),
          'frame', np.mean([r[3] for r in rows]),
          'steam', np.mean([r[4] for r in rows]))


def tail_sheet_3x(out, dest, start, end, step=4, tag=''):
    """3x tail/plant crop sheet, every `step` frames inclusive."""
    dest.mkdir(parents=True, exist_ok=True)
    cells = []
    for i in range(start, end + 1, step):
        if i >= len(out) or out[i] is None:
            continue
        c = cv2.resize(out[i][400:720, 880:1280], None, fx=3, fy=3,
                       interpolation=cv2.INTER_NEAREST)
        cv2.putText(c, f'n{i}', (8, 28), cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255), 2)
        cells.append(c)
        cv2.imwrite(str(dest / f'tail3x-n{i:03d}.png'), c)
    if not cells:
        return
    # wrap into rows of 4
    cols = 4
    rows = []
    for r in range(0, len(cells), cols):
        chunk = cells[r:r + cols]
        if len(chunk) < cols:
            pad = [np.zeros_like(cells[0])] * (cols - len(chunk))
            chunk = chunk + pad
        rows.append(np.hstack(chunk))
    sheet = np.vstack(rows)
    name = f'tail3x-sheet-n{start:03d}-{end:03d}-s{step}{tag}.png'
    cv2.imwrite(str(dest / name), sheet)


def face_edge_sheet_3x(out, dest, start, end, step=1):
    """3x crop of the right-cheek card edge (around x=830) and full face."""
    dest.mkdir(parents=True, exist_ok=True)
    faces, cheeks, ears = [], [], []
    for i in range(start, end + 1, step):
        if i >= len(out) or out[i] is None:
            continue
        face = cv2.resize(out[i][200:500, 600:940], None, fx=3, fy=3,
                          interpolation=cv2.INTER_NEAREST)
        cheek = cv2.resize(out[i][240:480, 760:920], None, fx=3, fy=3,
                           interpolation=cv2.INTER_NEAREST)
        ear = cv2.resize(out[i][140:280, 780:960], None, fx=3, fy=3,
                         interpolation=cv2.INTER_NEAREST)
        for im, lab in ((face, 'face'), (cheek, 'cheek'), (ear, 'Rear')):
            cv2.putText(im, f'n{i}', (8, 28), cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255), 2)
        faces.append(face); cheeks.append(cheek); ears.append(ear)
        cv2.imwrite(str(dest / f'face3x-n{i:03d}.png'), face)
        cv2.imwrite(str(dest / f'cheek3x-n{i:03d}.png'), cheek)
        cv2.imwrite(str(dest / f'Rear3x-n{i:03d}.png'), ear)
    def _sheet(cells, name, cols=4):
        if not cells:
            return
        rows = []
        for r in range(0, len(cells), cols):
            chunk = cells[r:r + cols]
            if len(chunk) < cols:
                chunk = chunk + [np.zeros_like(cells[0])] * (cols - len(chunk))
            rows.append(np.hstack(chunk))
        cv2.imwrite(str(dest / name), np.vstack(rows))
    _sheet(faces, f'face3x-sheet-n{start:03d}-{end:03d}.png')
    _sheet(cheeks, f'cheek3x-sheet-n{start:03d}-{end:03d}.png')
    _sheet(ears, f'Rear3x-sheet-n{start:03d}-{end:03d}.png')


def write_all_qc(out, src_u, dest, mp4=None):
    Y.VER = VER
    Y.write_qc(out, src_u, dest, mp4=mp4)
    seam_region_report(out, dest, src_u)
    mouth_seq(out, dest)
    tail_closeups(out, dest)
    blink_strips(out, dest)
    motion_strips(out, dest)
    edge_qc(out, src_u, dest)
    # required ae-ex crop sheets
    tail_sheet_3x(out, dest, 60, 120, step=4, tag='')
    # n90 is the flagged flick frame (not on the 60+4k grid)
    if 90 < len(out) and out[90] is not None:
        c = cv2.resize(out[90][400:720, 880:1280], None, fx=3, fy=3,
                       interpolation=cv2.INTER_NEAREST)
        cv2.putText(c, 'n90', (8, 28), cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255), 2)
        cv2.imwrite(str(dest / 'tail3x-n090.png'), c)
    tail_sheet_3x(out, dest, 90, 104, step=1, tag='-gaps')
    tail_sheet_3x(out, dest, 301, 361, step=4, tag='-cyc2')
    tail_sheet_3x(out, dest, 542, 580, step=4, tag='-cyc3')
    face_edge_sheet_3x(out, dest, 425, 440, step=1)
    face_edge_sheet_3x(out, dest, 132, 148, step=1)


def main():
    import argparse
    global MOUTH_REF_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument('--preview', action='store_true')
    ap.add_argument('--mouth-ref', default=None)
    args = ap.parse_args()
    if args.mouth_ref:
        MOUTH_REF_PATH = args.mouth_ref

    mp4 = P / f'{VER}-test-25s.mp4'
    if not args.preview and mp4.exists():
        raise SystemExit(f'refuse to overwrite {mp4}')

    for label, path, expect in (
        ('v41u-ex-mp4', P / 'v41u-ex-test-25s.mp4', V41U_SHA),
        ('v41u-ex-los', SRC_U, V41U_LOS_SHA),
        ('v41z-ex-mp4', P / 'v41z-ex-test-25s.mp4', V41Z_SHA),
        ('v41aa-ex-mp4', P / 'v41aa-ex-test-25s.mp4', V41AA_SHA),
        ('v41ab-ex-mp4', P / 'v41ab-ex-test-25s.mp4', V41AB_SHA),
        ('v41ac-ex-mp4', P / 'v41ac-ex-test-25s.mp4', V41AC_SHA),
        ('v41ad-ex-mp4', P / 'v41ad-ex-test-25s.mp4', V41AD_SHA),
        ('v41ae-ex-mp4', P / 'v41ae-ex-test-25s.mp4', V41AE_SHA),
        ('pcm', PCM, Hh.PCM_SHA),
    ):
        got = sha256(path)
        if got != expect:
            raise SystemExit(f'{label} sha changed: {got}')
    print('frozen shas ok', flush=True)
    print('mouth ref:', MOUTH_REF_PATH or '(none — hook is idle)', flush=True)

    print('loading ab-ex g1v3/g2v3 mattes (read-only)…', flush=True)
    pane = T.pane_masks().astype(np.float32)
    WORK.mkdir(exist_ok=True)
    for lab, path in (('g1v3', MATTE_G1), ('g2v3', MATTE_G2)):
        for name in ('cat-solid.npz', 'plant.npz', 'inpaint-gen.npz', 'chair-pocket.npz'):
            if not (path / name).exists():
                raise SystemExit(f'missing read-only matte {path / name}')
    c1 = np.load(MATTE_G1 / 'cat-solid.npz')['alpha']
    p1 = np.load(MATTE_G1 / 'plant.npz')['alpha']
    gi1 = np.load(MATTE_G1 / 'inpaint-gen.npz')['gen']
    h1 = np.load(MATTE_G1 / 'chair-pocket.npz')['alpha']
    c2 = np.load(MATTE_G2 / 'cat-solid.npz')['alpha']
    p2 = np.load(MATTE_G2 / 'plant.npz')['alpha']
    gi2 = np.load(MATTE_G2 / 'inpaint-gen.npz')['gen']
    h2 = np.load(MATTE_G2 / 'chair-pocket.npz')['alpha']
    gc.collect()
    print('repairing baked face hole in g1v3/g2v3 cat-solid (in memory)…', flush=True)
    c1 = repair_cat_stack(c1, gi1)
    c2 = repair_cat_stack(c2, gi2)
    gc.collect()

    cat_union = np.clip(np.maximum(c1[0].astype(np.float32), c2[0].astype(np.float32)) / 255.0, 0, 1)
    w_room, w_bulb = T.lamp_weights(pane, cat_union)
    QC.mkdir(parents=True, exist_ok=True)
    print('loading u-ex plates…', flush=True)
    src_u = Hh.load_span(SRC_U)

    if args.preview:
        idx = sorted(set(
            list(range(0, 8))
            + list(range(0, 40, 8))
            + list(range(46, 70))
            + list(range(60, 121))
            + list(range(132, 152))
            + list(range(210, 234))
            + list(range(236, 246))
            + list(range(241, 249))
            + list(range(260, 288, 2))
            + list(range(287, 310))
            + list(range(301, 362))
            + list(range(336, 385, 4))
            + list(range(424, 456))
            + list(range(476, 486))
            + list(range(482, 490))
            + list(range(500, 556, 2))
            + list(range(528, 548))
            + list(range(542, 581))
            + list(range(38, 56))
            + list(range(152, 172))
            + list(range(241 + 38, 241 + 56))
            + list(range(241 + 152, 241 + 172))
            + list(range(482 + 38, 482 + 56))
            + list(range(26, 36))  # head turn into hold A
            + list(range(50, 62))
            + list(range(66, 78))
            + [0, 20, 32, 64, 90, 92, 96, 100, 104, 114, 138, 352, 432, 599]
        ))
        print('preview frames', len(idx), flush=True)
        out = process(src_u, c1, c2, p1, p2, h1, h2, gi1, gi2,
                      w_room, w_bulb, preview=idx)
        dest = QC / 'preview'
        write_all_qc(out, src_u, dest, mp4=None)
        print('preview written', dest, flush=True)
        return

    out = process(src_u, c1, c2, p1, p2, h1, h2, gi1, gi2, w_room, w_bulb)
    y0, y1, x0, x1 = TITLE
    bad = [i for i, (a, b) in enumerate(zip(src_u, out))
           if not np.array_equal(a[y0:y1, x0:x1], b[y0:y1, x0:x1])]
    print('title mismatches' if bad else 'title core byte-identical',
          len(bad) if bad else '', flush=True)

    los = WORK / f'{VER}-span-890-915-lossless.mkv'
    print('writing lossless…', flush=True)
    Hh.write_lossless(out, los)
    print('encoding mp4…', flush=True)
    Hh.encode_mp4(out, mp4)
    print(VER, 'mp4', sha256(mp4), flush=True)
    print(VER, 'lossless', sha256(los), flush=True)
    try:
        print(VER, 'aac', aac_sha(mp4), flush=True)
    except Exception as e:
        print('aac hash failed', e, flush=True)
    shutil.copy2(mp4, WORK / f'{VER}-test-25s.mp4')
    shutil.copy2(Path(__file__), WORK / 'build_v41af_ex.py')
    dest = QC / 'encoded'
    write_all_qc(out, src_u, dest, mp4=mp4)
    # rest-hold vs aa if the plot data exists
    print('computing flow mags…', flush=True)
    mz = AA.flow_mags(out, src_u)
    np.savez(dest / 'flow-mags-af.npz', **{f'af_{k}': mz[k] for k in mz})
    def hold_frac(v, thr=0.15):
        return float(np.mean(v < thr))
    print('hold frac mag<0.15  af')
    for k in ('head', 'ears', 'chest', 'tail'):
        print(f'  {k:6s}  {hold_frac(mz[k]):.3f}')


if __name__ == '__main__':
    main()
