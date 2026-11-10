#!/usr/bin/env python3
"""v41ac-ex: same composite and organic warps as v41ab-ex, but replace
the offset sticker blinks with per-frame tracked eyelids (pupil/iris
+ catchlight on the already-warped plate). Lid follows each eye's
own upper contour, comes down from the top, squash toward the lower
lash, fur sampled from above that eye. If detection fails on either
eye, skip the blink rather than paint a one-sided or offset lid.

Does not edit or overwrite earlier builders or encodes. Reuses
v41ab-ex g1v3/g2v3 mattes read-only. Mouth-reference hook idle.
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

VER = 'v41ac-ex'
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
DRIFT_AMP_X, DRIFT_AMP_Y = 2.20, 1.70
TILT_DEG = 1.00
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

MOUTH_REF_PATH = os.environ.get('V41AC_MOUTH_REF') or os.environ.get('V41AB_MOUTH_REF') or os.environ.get('V41AA_MOUTH_REF') or None

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


def drift(local):
    th = 2.0 * np.pi * local / NCYCLE
    # both components DC-cancelled so displacement is 0 at the loop join
    dx = DRIFT_AMP_X * (
        0.70 * np.sin(th) + 0.30 * np.sin(2.0 * th + 1.1) - 0.30 * np.sin(1.1)
    )
    dy = DRIFT_AMP_Y * (
        0.65 * np.cos(th + 0.35) + 0.25 * np.sin(3.0 * th + 2.0)
        - 0.65 * np.cos(0.35) - 0.25 * np.sin(2.0)
    )
    return float(dx), float(dy)


def tilt_ang(local):
    th = 2.0 * np.pi * local / NCYCLE
    return np.deg2rad(TILT_DEG) * np.sin(th)  # 0 at local 0 and 241


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
        # scale about (700, 530); fade with chest+head so paws stay planted later
        body = np.clip(wt['chest'] + 0.55 * wt['head'], 0, 1)
        dx = dx + body * (sc * (Xg - 700.0))
        dy = dy + body * (sc * (Yg - 530.0))

    # head: lagged breath, wander, 1° tilt, anticipation
    dy = dy + wt['head'] * (-HEAD_BREATH_AMP * k_h + drift_y + ant_dy)
    dx = dx + wt['head'] * drift_x
    if abs(tang) > 1e-8:
        dxt, dyt = rotation_disp(*TILT_PIVOT, tang)
        dx = dx + wt['head'] * dxt
        dy = dy + wt['head'] * dyt

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

def compose_layered(uex, gen, a_cat, a_plant, a_chair, ph, ph0, w_room, w_bulb):
    a_c = np.clip(a_cat, 0, 1)
    a_p = np.clip(a_plant, 0, 1)
    a_h = np.clip(np.maximum(a_chair, corner_gate()), 0, 1)
    a_all = np.clip(np.maximum(np.maximum(a_c, a_p), a_h), 0, 1)
    gen_f = Y.residual_flicker(gen, w_room, w_bulb, ph, ph0)
    gen_f = Y.color_match(gen_f, uex, a_all)
    # cat (and chair pockets) first, plant wins on top
    a_under = np.clip(np.maximum(a_c, a_h) * (1.0 - a_p), 0, 1)
    out = uex.astype(np.float32) * (1.0 - a_under[..., None]) + gen_f * a_under[..., None]
    out = out * (1.0 - a_p[..., None]) + gen_f * a_p[..., None]
    off = a_all < 0.02
    out[off] = uex[off].astype(np.float32)
    y0, y1, x0, x1 = TITLE
    out[y0:y1, x0:x1] = uex[y0:y1, x0:x1]
    return np.clip(out, 0, 255).astype(np.uint8)


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
    # unwarped corner coverage (chair pockets + hole fills), minus the moving cat
    a_stat = np.clip(np.maximum(a_p, a_chair * (1.0 - a_c)), 0, 1)
    mx = (Xg - dx).astype(np.float32)
    my = (Yg - dy).astype(np.float32)
    pl_w = cv2.remap(plate, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    a_w = cv2.remap(a_c * (1.0 - a_p), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    a_w = np.clip(a_w, 0, 1)
    dil = cv2.dilate((a_c > 0.08).astype(np.uint8),
                     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    dil = cv2.GaussianBlur(dil.astype(np.float32), (0, 0), 2.0)
    bg = plate.astype(np.float32) * (1.0 - dil[..., None]) + u.astype(np.float32) * dil[..., None]
    out = bg * (1.0 - a_w[..., None]) + pl_w.astype(np.float32) * a_w[..., None]
    # unwarped coverage + plant back on top (leaves keep a clean edge)
    out = out * (1.0 - a_stat[..., None]) + plate.astype(np.float32) * a_stat[..., None]
    off = (a_w < 0.02) & (a_stat < 0.02) & (dil < 0.02)
    out[off] = plate[off].astype(np.float32)
    y0, y1, x0, x1 = TITLE
    out[y0:y1, x0:x1] = plate[y0:y1, x0:x1]
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
def process(src_u, c1, c2, p1, p2, h1, h2, gi1, gi2, w_room, w_bulb, preview=None):
    want = set(preview) if preview else set(range(NFR))
    out = [None] * NFR
    global _eye_priors
    _eye_priors = dict(EYE_PRIOR)
    start1 = (gi1[0].astype(np.float32), c1[0]/255.0, p1[0]/255.0, h1[0]/255.0)
    start2 = (gi2[0].astype(np.float32), c2[0]/255.0, p2[0]/255.0, h2[0]/255.0)
    ph0 = Hh.phase_of(0)
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
        plate = compose_layered(src_u[i], gen, a_c, a_p, a_h, ph, ph0, w_room, w_bulb)
        dx, dy = warp_field(i)
        gate = cv2.dilate((np.maximum(a_c, a_h) > 0.05).astype(np.uint8),
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
        gate = cv2.GaussianBlur(gate.astype(np.float32), (0, 0), 4.0)
        dx, dy = dx * gate, dy * gate
        im = compose_warped(plate, src_u[i], a_c, a_p, a_h, dx, dy)
        im = apply_blink_tracked(im, blink_k(local), i)
        im = apply_mouth_hook(im, i)
        out[i] = im
        if preview or i % 25 == 0 or w is not None or (use2 and 188 <= local <= 204):
            extra = f' seam_w={w:.3f}' if w is not None else ''
            print(f'  n{i} local={local} k={K_BREATH[local]:.2f} '
                  f'dy_chest={-CHEST_AMP*K_BREATH[local]:.2f} blink={blink_k(local):.2f}{extra}',
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
        mid = 13
        r1 = np.hstack(cells[:mid])
        r2 = np.hstack(cells[mid:])
        if r2.shape[1] < r1.shape[1]:
            r2 = np.hstack([r2, np.zeros((r2.shape[0], r1.shape[1] - r2.shape[1], 3), np.uint8)])
        cv2.imwrite(str(dest / 'meow-seq-432-every.png'), np.vstack([r1, r2]))


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
    # every blink frame in range, plus one open frame on each side
    return [
        ('A0', list(range(47, 59))),    # local 48-58
        ('B0', list(range(213, 225))),  # local 214-223
        ('A1', list(range(288, 301))),  # n289-299
        ('B1', list(range(454, 466))),  # n455-464
        ('A2', list(range(529, 542))),  # n530-540
    ]


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


def write_all_qc(out, src_u, dest, mp4=None):
    Y.VER = VER
    Y.write_qc(out, src_u, dest, mp4=mp4)
    seam_region_report(out, dest, src_u)
    mouth_seq(out, dest)
    tail_closeups(out, dest)
    blink_strips(out, dest)
    motion_strips(out, dest)
    edge_qc(out, src_u, dest)


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

    cat_union = np.clip(np.maximum(c1[0].astype(np.float32), c2[0].astype(np.float32)) / 255.0, 0, 1)
    w_room, w_bulb = T.lamp_weights(pane, cat_union)
    QC.mkdir(parents=True, exist_ok=True)
    print('loading u-ex plates…', flush=True)
    src_u = Hh.load_span(SRC_U)

    if args.preview:
        idx = sorted(set(
            list(range(0, 40, 8))
            + list(range(46, 60))
            + list(range(70, 86))
            + list(range(106, 125))
            + list(range(132, 145))
            + list(range(212, 226))
            + list(range(236, 246))
            + list(range(287, 301))
            + list(range(345, 362))
            + list(range(424, 445))
            + list(range(453, 466))
            + list(range(476, 486))
            + list(range(528, 542))
            + [0, 20, 32, 599]
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
    shutil.copy2(Path(__file__), WORK / 'build_v41ac_ex.py')
    dest = QC / 'encoded'
    write_all_qc(out, src_u, dest, mp4=mp4)
    # rest-hold vs aa if the plot data exists
    print('computing flow mags…', flush=True)
    mz = AA.flow_mags(out, src_u)
    np.savez(dest / 'flow-mags-ac.npz', **{f'ac_{k}': mz[k] for k in mz})
    def hold_frac(v, thr=0.15):
        return float(np.mean(v < thr))
    print('hold frac mag<0.15  ac')
    for k in ('head', 'ears', 'chest', 'tail'):
        print(f'  {k:6s}  {hold_frac(mz[k]):.3f}')


if __name__ == '__main__':
    main()
