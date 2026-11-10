#!/usr/bin/env python3
"""v41j: v41i + mullion/ear strip, lamp-rim halo, softer gap glimpses, a touch more dust.

v41i notes follow.
v41i: fixes v41h honest leftovers.

Built from frozen v41d lossless, reusing v41h/v41f-r2/v41g pieces.
- Window: static-background (masked temporal median) landscape -> dark tree
  silhouettes, no light halos; moving flakes isolated as frame - background
  (clean, soft, not tophat scratches); dark sky with a rose-gold horizon lip.
- Tail: edge-decontaminated sprite (no dark fringe), choked alpha, tip/cream
  colour matched to Mimi's graded haunch fur, hook opened at idle.
- Room: extra lamp-centred falloff (global luma ~55 -> ~49), warm flicker
  enveloped by the v28 3-cycle pulse (no stacking pump).
- Dust: fewer, softer motes on fixed orbits that never cross fur/face/glass
  (no pop-in/out).
Never overwrites earlier versions.
"""
from __future__ import annotations
import sys
from pathlib import Path
import cv2
import numpy as np

P = Path('/workspace/lofi-v41-meow')
sys.path.insert(0, str(P))
import build_v41h as Hh
import build_v41c as C
import build_v41f_r2 as R
import build_v41g as G

VER = 'v41j-r2'
QC = P / 'qc' / VER
for d in ('preview', 'encoded'):
    (QC / d).mkdir(parents=True, exist_ok=True)
WORK = P / 'v41j'
WORK.mkdir(exist_ok=True)
SPRITE = P / 'v41j' / 'tail-sprite.png'
NFR, W, H, FPS, NCYCLE = 600, 1280, 720, 24, 241
TITLE = Hh.TITLE
Y, X = R.Y, R.X
sstep, luma, hex_bgr, phase_of = Hh.sstep, Hh.luma, Hh.hex_bgr, Hh.phase_of
WIN = (0, 420, 250, 872)  # r2: was 850, starved the strip by the right post  # y0,y1,x0,x1

V41H_MP4 = P / 'v41i-test-25s.mp4'  # previous logged best must stay frozen
V41H_SHA = '77bc1bbf7a3b0f38c63497542aaa164a5dbf0a657284b895811d100c8ca86811'

# tunables (overridden by later rounds)
CFG = dict(
    room_far_dark=0.16,     # extra multiply away from lamp
    cat_body_dark=0.07,
    flick_amp=0.010,
    tail_open=11.0,         # idle opening of the hook (px at tip)
    tail_lift=-5.0,
    tail_curl=6.0,
    tail_gain=0.93,         # vs graded haunch white
    flake_gain=1.0,
    flake_cap=46.0,
    tree_contrast=0.075,
    tree_base=9.0,
)


# ---------------- pane + leaf masks ----------------
LEAF_ZONE = np.clip(np.maximum.reduce([
    sstep((345.0 - X) / 8.0) * sstep((Y - 215.0) / 8.0),
    sstep((Y - 325.0) / 8.0) * sstep((455.0 - X) / 8.0),
    sstep((98.0 - Y) / 4.0) * np.maximum(sstep((335.0 - X) / 6.0), sstep((X - 790.0) / 6.0)),
]), 0, 1)


# Glass runs to the top of the frame (y=0), right to the post (x~852) and down
# to the sill (y~411); the v41f-h polygons stopped at y=90 / x=828 / y=396 and
# left hard day-sky strips. Extensions are gated to bluish (glass) pixels of
# the static background so wood / ivy / lamp are never dusk-graded.
PANE_L_OLD = np.array([[266, 90], [560, 90], [558, 394], [268, 396]], np.int32)
PANE_R_OLD = np.array([[600, 90], [828, 92], [826, 396], [600, 396]], np.int32)
PANE_L_EXT = np.array([[262, 0], [564, 0], [564, 412], [264, 412]], np.int32)
PANE_R_EXT = np.array([[593, 0], [853, 0], [853, 412], [593, 412]], np.int32)


def static_masks(bg):
    old = np.zeros((H, W), np.uint8)
    cv2.fillPoly(old, [PANE_L_OLD, PANE_R_OLD], 1)
    ext = np.zeros((H, W), np.uint8)
    cv2.fillPoly(ext, [PANE_L_EXT, PANE_R_EXT], 1)
    b = bg.astype(np.float32)
    B, G, R = b[..., 0], b[..., 1], b[..., 2]
    Yb = luma(b)
    blu = ~(((R - B) > 25) & (Yb < 135)) & (Yb > 40)
    blu = cv2.morphologyEx(blu.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    # v41j: above the old polygons, between the lamp and the right ivy, the
    # extension is all glass (olive-shaded conifers failed the colour gate).
    sure = np.zeros((H, W), np.uint8)
    sure[0:93, 335:561] = 1
    sure[0:93, 596:790] = 1
    copper = ((R - G) > 18) & ((R - B) > 40)
    copper = cv2.dilate(copper.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    sure = (sure.astype(bool) & ~copper).astype(np.uint8)
    panes = np.maximum(old, ext * np.maximum(blu, sure)).astype(np.float32)
    panes = cv2.GaussianBlur(panes, (0, 0), 1.4)
    # Indoor leaves over glass (static): green-over-blue test inside the zones
    # where desk plants / ivy overlap the window. Teal snowy conifers are
    # B>=G, so they are never 'leaf'.
    hsv = cv2.cvtColor(np.clip(b, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    h, sat, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    # leaf = saturated green-over-blue, or dark non-blue; then holes filled
    g1 = ((G - B) > 6) & (sat > 70) & (h > 23) & (h < 90)
    g2 = (Yb < 70) & ((B - R) < 20) & (Yb > 1)
    gz = ((g1 | g2) & (LEAF_ZONE > 0.5)).astype(np.uint8)
    gz = cv2.morphologyEx(gz, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    gz = cv2.morphologyEx(gz, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(gz, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(gz, cnts, -1, 1, -1)
    green = gz.astype(np.float32)
    strict = sstep((h - 24) / 6) * sstep((58 - h) / 6) * sstep((sat - 95) / 20) * sstep((v - 35) / 20)
    leaf = np.maximum(green * LEAF_ZONE, strict)
    leaf = (leaf > 0.5).astype(np.uint8)
    leaf = cv2.morphologyEx(leaf, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    leaf = cv2.GaussianBlur(leaf.astype(np.float32), (0, 0), 0.7)
    return panes, leaf


# ---------------- static window background ----------------
def window_background(src, mattes):
    y0, y1, x0, x1 = WIN
    idx = list(range(0, NFR, 3))
    stack = np.empty((len(idx), y1 - y0, x1 - x0, 3), np.float32)
    for j, i in enumerate(idx):
        f = src[i][y0:y1, x0:x1].astype(np.float32)
        a = mattes[i][y0:y1, x0:x1].astype(np.float32)
        a = a / 255.0 if a.max() > 1.5 else a
        cov = cv2.dilate((a > 0.15).astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
        f[cov] = np.nan
        stack[j] = f
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        bg = np.nanmedian(stack, axis=0)
    valid = np.isfinite(bg[..., 0])
    cnt = np.isfinite(stack[..., 0]).sum(0)
    full = np.zeros((H, W, 3), np.float32)
    vfull = np.zeros((H, W), np.float32)
    # pixels the matte covers in every sample: plain (unmasked) median, never 0
    raw = np.median(np.stack([src[i][y0:y1, x0:x1] for i in idx], 0), axis=0).astype(np.float32)
    bgc = np.where(np.isfinite(bg), bg, raw)
    full[y0:y1, x0:x1] = bgc
    vfull[y0:y1, x0:x1] = (valid & (cnt >= 12)).astype(np.float32)
    vfull = cv2.GaussianBlur(cv2.erode(vfull, np.ones((5, 5), np.uint8)), (0, 0), 2.0)
    return full, vfull


def dusk_plate_i(wb):
    """Deep indigo zenith -> dusk blue -> thin rose-gold horizon lip behind the treeline."""
    stops_y = np.array([0, 30, 60, 90, 115, 135, 150, 162, 172, 182, 200, 260, 420], np.float32)
    k = 0.88 + 0.12 * wb
    cols = np.stack([
        hex_bgr('070A24'), hex_bgr('0A0E2E'), hex_bgr('10153A'), hex_bgr('182046'),
        hex_bgr('262650'), hex_bgr('3E2C56'),
        hex_bgr('6E3C5A') * k, hex_bgr('9A5A5C') * k, hex_bgr('B9845F') * k, hex_bgr('9E7060') * k,
        hex_bgr('5A3A4A'), hex_bgr('221C3C'), hex_bgr('181634'),
    ], 0)
    c = np.zeros((H, W, 3), np.float32)
    for ch in range(3):
        c[..., ch] = np.interp(Y, stops_y, cols[:, ch])
    return cv2.GaussianBlur(c, (0, 0), 3.0)


_FIX = {}


def lamp_fixture_i(frame, Lamp, panes):
    """Outside the glass: v41f-r2 geometric + colour fixture. Inside the glass:
    a static, smoothed shade mask (R-B > 110 = lit cream underside / copper),
    so warm-lit snow under the rim is graded instead of kept as 'lamp'."""
    ph = np.clip(panes * 1.5, 0, 1)
    out_glass = np.maximum(Lamp['fixture'], R.lamp_color_mask(frame)) * (1.0 - ph)
    if 'shade' not in _FIX:
        b = _FIX['bg'].astype(np.float32)
        geo = cv2.dilate((Lamp['fixture'] > 0.05).astype(np.uint8), np.ones((15, 15), np.uint8))
        hood = (Lamp['hood'] > 0.5) & (luma(b) < 140) & ((b[..., 2] - b[..., 0]) > 25)
        m = ((((b[..., 2] - b[..., 0]) > 110) | hood) & (geo > 0)).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnts = [c for c in cnts if cv2.contourArea(c) > 400]
        m2 = np.zeros_like(m)
        cv2.drawContours(m2, cnts, -1, 1, -1)
        m2 = cv2.morphologyEx(m2, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))  # r2: no rim drips
        _FIX['shade'] = cv2.GaussianBlur(m2.astype(np.float32), (0, 0), 1.0)
    if 'ring' not in _FIX:
        sh = (_FIX['shade'] > 0.5).astype(np.uint8)
        ring = cv2.dilate(sh, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) - sh
        ext = np.zeros((H, W), np.uint8)
        cv2.fillPoly(ext, [PANE_L_EXT], 1)
        b = _FIX['bg'].astype(np.float32)
        brightish = (luma(b) > 95).astype(np.uint8)
        _FIX['ring'] = cv2.GaussianBlur((ring * ext * brightish).astype(np.float32), (0, 0), 1.0)
    return np.clip(np.maximum(out_glass * (1.0 - _FIX['ring']), _FIX['shade'] * ph), 0, 1)


def grade_window_i(frame, bg, bgv, panes, cat_p, fixture, leaves, steam, title, wb, cat_a=None):
    open_ = 1.0 - cat_p
    if cat_a is not None:
        # The ~6 px cat_protect band left a pale-blue strip of day sky above the
        # head. Grade it where the pixel is clearly bluish (sky/snow), never fur.
        fb = frame.astype(np.float32)
        blu = sstep(((fb[..., 0] - fb[..., 2]) + 14.0) / 12.0)
        hard = cv2.GaussianBlur(np.clip(cat_a, 0, 1), (0, 0), 0.8)
        o1 = (1.0 - hard) * blu * 0.95
        # neutral snow right beside the matte (B-R ~ -25) also leaked; outside a
        # 3 px dilation of the matte, grade anything that is not ginger fur.
        near = cv2.dilate((cat_a > 0.05).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))).astype(np.float32)
        notginger = sstep(((fb[..., 0] - fb[..., 2]) + 45.0) / 12.0)
        o2 = (1.0 - near) * notginger * 0.95
        # v41j: the strip between the mullion and the left ear (x~592-606):
        # soft matte there is mostly glass; grade non-ginger pixels by (1-alpha).
        strip = sstep((X - 589.0) / 2.0) * sstep((610.0 - X) / 3.0) * sstep((Y - 140.0) / 6.0) * sstep((320.0 - Y) / 6.0)
        o3 = strip * notginger * np.clip(1.0 - 1.6 * cat_a, 0, 1) * 0.95
        o2 = np.maximum(o2, o3)
        open_ = np.maximum(open_, cv2.GaussianBlur(np.maximum(o1, o2), (0, 0), 0.8))
    glass = panes * open_ * (1.0 - fixture) * (1.0 - leaves) * (1.0 - 0.7 * steam) * (1.0 - title)
    glass = np.clip(glass, 0, 1)
    img = frame.astype(np.float32)
    # background: static where valid, current frame where the cat ever sits
    base = bg * bgv[..., None] + img * (1.0 - bgv)[..., None]
    t = np.clip(Y / 411.0, 0, 1)
    blow = cv2.GaussianBlur(base, (0, 0), 3.0)
    Bl, Gl, Rl = blow[..., 0], blow[..., 1], blow[..., 2]
    Yb = luma(blow)
    Yf0 = luma(base)
    mu = cv2.GaussianBlur(Yf0, (0, 0), 3.0)
    var = cv2.GaussianBlur((Yf0 - mu) ** 2, (0, 0), 3.0)
    textured = sstep((np.sqrt(np.clip(var, 0, None)) - 5.0) / 8.0)
    warm = sstep(((Rl - Bl) - 4.0) / 22.0)
    bright = sstep((Yb - 150.0) / 30.0)
    upper = sstep((0.48 - t) / 0.08)
    sky = upper * bright * np.maximum(warm, 0.6 * (1 - textured))
    sky = cv2.GaussianBlur(np.clip(sky, 0, 1), (0, 0), 2.2)
    sky = np.clip(sky * 1.15, 0, 1)
    # landscape luma from non-sky pixels only (normalized conv -> no bright halo)
    ns = np.clip(1.0 - sky, 0, 1)
    num = cv2.GaussianBlur(base * ns[..., None], (0, 0), 2.2)
    den = cv2.GaussianBlur(ns, (0, 0), 2.2)[..., None] + 1e-3
    lowl = num / den
    Yl = luma(lowl)
    # detail (static painted texture), kept mild
    det = luma(base) - luma(cv2.GaussianBlur(base, (0, 0), 1.2))
    # trees (upper/mid): dark silhouettes, snow faintly visible
    up_w = sstep((0.74 - t) / 0.12)
    Yt = CFG['tree_base'] + 1.5 * wb + np.clip(Yl - 70.0, 0, None) * CFG['tree_contrast']
    Yg = (17.0 + 3.0 * wb) + (Yl - 90.0) * 0.24  # lower ground snow
    Y2 = np.clip(Yt * up_w + Yg * (1 - up_w), 4, 46)
    scale = np.clip(Y2 / (Yl + 1e-3), 0.04, 0.42)
    tint_tree = np.array([1.30, 1.02, 0.72], np.float32)
    tint_gr = np.array([1.12, 0.98, 0.82], np.float32)
    tint = tint_tree[None, None] * up_w[..., None] + tint_gr[None, None] * (1 - up_w)[..., None]
    land = lowl * scale[..., None] * tint
    land = land + (det * 0.22 * (0.5 + 0.5 * (1 - up_w)))[..., None]
    # afterglow bounce on the ground snow, rim light on the treeline
    ag = sstep((t - 0.72) / 0.12) * np.clip(Yl / 140.0, 0.1, 1.0)
    bounce = (0.050 + 0.025 * wb) * ag
    land = land + hex_bgr('C47A6A') * bounce[..., None] + hex_bgr('D4A565') * (0.3 * bounce)[..., None]
    edge = np.clip(cv2.GaussianBlur(sky, (0, 0), 2.0) - sky, 0, 1) * sstep((0.50 - t) / 0.10)
    land = land + hex_bgr('B4675A') * (0.10 * edge)[..., None]
    plate = dusk_plate_i(wb)
    new_low = plate * sky[..., None] + land * (1.0 - sky)[..., None]
    # moving flakes = current frame minus static background (keeps snow motion exactly)
    d = luma(img) - luma(base)
    d = np.clip(d - 3.0, 0, None)
    d = cv2.GaussianBlur(d, (0, 0), 0.6)
    cap = CFG['flake_cap']
    fl = cap * (1.0 - np.exp(-d * CFG['flake_gain'] / cap))
    fl = fl * sstep((bgv - 0.90) / 0.10)
    flake_col = np.array([1.02, 0.99, 0.94], np.float32)
    new_pane = new_low + fl[..., None] * flake_col
    # in never-valid (cat-adjacent) pixels fall back to v41h grade there
    new_pane = np.clip(new_pane, 0, 255)
    g = glass[..., None]
    return img * (1.0 - g) + new_pane * g, glass


# ---------------- room: falloff + coordinated flicker ----------------
def v28_pulse(ph):
    return float((0.5 - 0.5 * np.cos(2 * np.pi * 3 * ph / NCYCLE)) ** 2)


def flicker(ph):
    env = 0.35 + 0.65 * v28_pulse(ph)
    v = (0.55 * np.sin(2 * np.pi * 7 * ph / NCYCLE + 0.3)
         + 0.30 * np.sin(2 * np.pi * 13 * ph / NCYCLE + 1.1)
         + 0.15 * np.sin(2 * np.pi * 19 * ph / NCYCLE + 2.0))
    return float(env * v)


_ROOM = {}


def room_masks():
    if _ROOM:
        return _ROOM
    pool = R.lamp_pool()
    lamp = R.lamp_masks()
    face = G.face_protect()
    far = np.clip(1.0 - pool, 0, 1)
    _ROOM.update(pool=pool, lamp=lamp, face=face, far=far,
                 near=cv2.GaussianBlur(np.clip(np.maximum(pool, lamp['protect']), 0, 1), (0, 0), 6.0))
    return _ROOM


def room_extra(img, cat_a, glass, ph):
    m = room_masks()
    cat_s = cv2.GaussianBlur(np.clip(cat_a, 0, 1), (0, 0), 6.0)
    face = m['face']
    dark = CFG['room_far_dark'] * m['far'] * (1.0 - cat_s) * (1.0 - glass)
    dark += CFG['cat_body_dark'] * cat_s * (1.0 - face) * (0.4 + 0.6 * m['far'])
    dark *= (1.0 - m['lamp']['protect'])
    gain = 1.0 - cv2.GaussianBlur(np.clip(dark, 0, 0.5), (0, 0), 4.0)
    out = img * gain[..., None]
    # warm flicker on pool + fixture, enveloped by the v28 pulse (zero-mean)
    fk = CFG['flick_amp'] * flicker(ph)
    wmask = np.clip(0.7 * m['near'] + 0.6 * m['lamp']['protect'], 0, 1) * (1.0 - 0.8 * glass)
    warmv = np.array([0.55, 0.85, 1.0], np.float32)
    out = out * (1.0 + fk * wmask[..., None] * warmv)
    return out


# ---------------- dust: fixed safe orbits, softer ----------------
def build_motes(mattes):
    union = np.zeros((H, W), np.float32)
    for i in range(0, NFR, 4):
        a = mattes[i].astype(np.float32)
        a = a / 255.0 if a.max() > 1.5 else a
        union = np.maximum(union, a)
    unsafe = cv2.dilate((union > 0.2).astype(np.uint8), np.ones((31, 31), np.uint8)).astype(np.float32)
    unsafe = np.maximum(unsafe, (G.face_protect() > 0.05).astype(np.float32))
    unsafe = np.maximum(unsafe, (G.pane_solid() > 0.05).astype(np.float32))
    y0, y1, x0, x1 = TITLE
    unsafe[max(0, y0 - 20):y1 + 20, max(0, x0 - 20):x1 + 20] = 1
    keep = []
    for m in G.MOTES:
        ok = True
        for ph in range(0, NCYCLE, 2):
            x, y = mote_pos(m, ph)
            xi, yi = int(round(x)), int(round(y))
            if xi < 12 or yi < 12 or xi >= W - 12 or yi >= H - 12 or unsafe[yi, xi] > 0:
                ok = False
                break
        if ok:
            mm = dict(m)
            mm['peak'] = m['peak'] * (0.75 if m['cone'] else 1.0)
            keep.append(mm)
    # sparse: keep at most 14
    keep = keep[:14]
    return keep, unsafe


def mote_pos(m, ph):
    k = m['k']
    tp = 2 * np.pi
    x = m['x0'] + m['rx'] * np.sin(tp * k * ph / NCYCLE + m['p1'])
    y = m['y0'] + m['ry'] * np.cos(tp * k * ph / NCYCLE + m['p2'])
    x += m['jx'] * np.sin(tp * (k + 3) * ph / NCYCLE + m['p3'])
    y += m['jy'] * np.cos(tp * (k + 2) * ph / NCYCLE + m['p1'])
    return x, y


SOFT_LAYER = {0: dict(core=3.4, sig=1.6, halo=3.6), 1: dict(core=2.2, sig=1.0, halo=0.0),
              2: dict(core=1.4, sig=1.4, halo=0.0)}
_KC = {}


def apply_dust_i(img, motes, unsafe_soft, ph):
    dust = np.zeros((H, W, 3), np.float32)
    for m in motes:
        key = (m['layer'], m['cone'], round(m['peak'], 3))
        if key not in _KC:
            L = SOFT_LAYER[m['layer']]
            col = np.array([0xB8, 0xDC, 0xF6] if m['cone'] else [0xA8, 0xC8, 0xD8], np.float32)
            _KC[key] = G._kernel(L['core'], L['sig'], L['halo'] if m['cone'] else 0.0, col, m['peak'])
        ker, rad = _KC[key]
        x, y = mote_pos(m, ph)
        # sub-pixel placement (no 1px hopping)
        xi, yi = int(np.floor(x)), int(np.floor(y))
        fx, fy = x - xi, y - yi
        Mt = np.float32([[1, 0, fx], [0, 1, fy]])
        k2 = cv2.warpAffine(ker, Mt, (ker.shape[1], ker.shape[0]), flags=cv2.INTER_LINEAR)
        scale = 0.12 + 0.88 * G.cone_w(x, y) if m['cone'] else 1.0
        # slow twinkle-free breathing, integer cycles
        scale *= 0.85 + 0.15 * np.sin(2 * np.pi * (m['k'] + 1) * ph / NCYCLE + m['p2'])
        dust[yi - rad:yi + rad + 1, xi - rad:xi + rad + 1] += k2 * scale
    dust *= (1.0 - unsafe_soft[..., None])
    add = np.clip(dust * 0.80, 0, 255)
    base = img.astype(np.float32)
    out = 255.0 - (255.0 - base) * (1.0 - add / 255.0)
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------- tail ----------------
def fix_sprite(sp, target_white):
    a = sp[:, :, 3].astype(np.float32) / 255.0
    rgb = sp[:, :, :3].copy()
    # decontaminate the edge band: inpaint colours from the solid interior outward
    solid = (a > 0.93).astype(np.uint8)
    solid = cv2.erode(solid, np.ones((3, 3), np.uint8))
    band = ((a > 0.003) & (solid == 0)).astype(np.uint8) * 255
    band = cv2.dilate(band, np.ones((3, 3), np.uint8))
    rgb_f = cv2.inpaint(rgb, band, 6, cv2.INPAINT_TELEA).astype(np.float32)
    w = np.clip((0.93 - a) / 0.5, 0, 1)[..., None]
    rgb2 = rgb.astype(np.float32) * (1 - w) + rgb_f * w
    # choke alpha slightly (removes the matte's dark rim), keep wisps
    a2 = np.clip((a - 0.10) / 0.85, 0, 1) ** 1.15
    # colour-match the cream to Mimi's graded white fur
    Yv = luma(rgb2)
    sat = rgb2.max(2) - rgb2.min(2)
    cream = sstep((Yv - 88.0) / 25.0) * sstep((115.0 - sat) / 40.0) * (a2 > 0.3)
    cur = (rgb2 * cream[..., None]).reshape(-1, 3).sum(0) / (cream.sum() + 1e-6)
    return rgb2, a2, cream, cur


_TAIL = {}


def prep_tail_padded(sprite_path, PAD=90):
    sp = cv2.imread(str(sprite_path), cv2.IMREAD_UNCHANGED)
    assert sp is not None and sp.shape[2] == 4
    # v41h rotated inside the unpadded sprite canvas, which clipped the
    # tip's outer edge into a straight vertical cut. Pad first.
    sp = cv2.copyMakeBorder(sp, PAD, PAD, PAD, PAD, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
    a = sp[:, :, 3].astype(np.float32) / 255.0
    rgb = sp[:, :, :3].astype(np.float32)
    # extra settle into the dusk room
    rgb = rgb * 0.90
    # Hand-traced medial axis on tail-sprite.png (360x384). The automatic
    # ridge walk folded back on itself and turned the warp into a starburst.
    poly_s = np.array([
        [25, 305], [40, 255], [55, 210], [75, 170], [110, 130],
        [155, 100], [200,  78], [245,  68], [285,  78], [312, 108],
    ], np.float32) + PAD
    base = poly_s[0].copy()
    tip = poly_s[-1].copy()

    # scale
    sh, sw = sp.shape[:2]
    nh, nw = int(round(sh * Hh.TAIL_SCALE)), int(round(sw * Hh.TAIL_SCALE))
    rgb_s = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    a_s = cv2.resize(a, (nw, nh), interpolation=cv2.INTER_AREA)
    poly_s = poly_s * Hh.TAIL_SCALE
    base_s = base * Hh.TAIL_SCALE
    tip_s = tip * Hh.TAIL_SCALE

    # rotate around base so rest tip aims at Hh.TIP_W
    v_src = tip_s - base_s
    v_dst = Hh.TIP_W - Hh.BASE_W
    ang = np.degrees(np.arctan2(v_dst[1], v_dst[0]) - np.arctan2(v_src[1], v_src[0]))
    M = cv2.getRotationMatrix2D(tuple(base_s), -ang, 1.0)
    rgb_r = cv2.warpAffine(rgb_s, M, (nw, nh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    a_r = cv2.warpAffine(a_s, M, (nw, nh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    # transform polyline
    ones = np.ones((len(poly_s), 1), np.float32)
    pr = (M @ np.hstack([poly_s, ones]).T).T
    br = (M @ np.array([base_s[0], base_s[1], 1.0])).ravel()
    # world translation
    ox, oy = Hh.BASE_W[0] - br[0], Hh.BASE_W[1] - br[1]
    rest_poly = pr + np.array([ox, oy], np.float32)
    # Half-width along the rest poly, in world/sprite-local pixels (scale already applied).
    dt = cv2.distanceTransform((a_r > 0.25).astype(np.uint8), cv2.DIST_L2, 3)
    widths = []
    for q in pr:  # sprite-local after rotation
        x, y = int(round(q[0])), int(round(q[1]))
        x = int(np.clip(x, 0, nw - 1)); y = int(np.clip(y, 0, nh - 1))
        widths.append(float(dt[y, x]))
    widths = np.array(widths, np.float32)
    widths = np.maximum(widths, 8.0)
    # World-space rest sprite (identity composite when undeformed).
    world_rgb = np.zeros((H, W, 3), np.float32)
    world_a = np.zeros((H, W), np.float32)
    x0, y0 = int(np.floor(ox)), int(np.floor(oy))
    # rgb_r is already in a canvas of size (nh,nw) whose (0,0) is sprite-local
    # world pos = sprite-local + (ox,oy)
    # use rounded ox,oy — subpixel is handled by having ox,oy floats on the poly only
    def _paste(dst, src, x0, y0):
        h, w = src.shape[:2]
        y1, x1 = y0 + h, x0 + w
        sy0 = sx0 = 0
        sy1, sx1 = h, w
        if y0 < 0:
            sy0 = -y0; y0 = 0
        if x0 < 0:
            sx0 = -x0; x0 = 0
        if y1 > dst.shape[0]:
            sy1 -= (y1 - dst.shape[0]); y1 = dst.shape[0]
        if x1 > dst.shape[1]:
            sx1 -= (x1 - dst.shape[1]); x1 = dst.shape[1]
        if y1 > y0 and x1 > x0:
            dst[y0:y1, x0:x1] = src[sy0:sy1, sx0:sx1]
    _paste(world_rgb, rgb_r, int(round(ox)), int(round(oy)))
    _paste(world_a, a_r, int(round(ox)), int(round(oy)))
    return {
        'rgb': rgb_r, 'a': a_r, 'ox': ox, 'oy': oy,
        'rest': rest_poly, 'nw': nw, 'nh': nh,
        'widths': widths,
        'world_rgb': world_rgb, 'world_a': world_a,
    }



def prep_tail(graded_ref):
    sp = cv2.imread(str(SPRITE), cv2.IMREAD_UNCHANGED)
    hw = graded_ref[600:660, 960:1000].reshape(-1, 3).astype(np.float32).mean(0)
    rgb2, a2, cream, cur = fix_sprite(sp, hw)
    # global sprite gain used in v41h was 0.90 inside _prep_tail; we write a
    # pre-corrected sprite and neutralise that 0.90 by dividing here.
    target = hw * CFG['tail_gain']
    # per-channel gain on cream weighted region, soft so ginger keeps its hue
    gain = target / (cur * 0.90 + 1e-6)
    # non-cream parts: match overall level to graded ginger with a milder gain
    lum_gain = float(luma(target[None, None])[0, 0] / (luma((cur * 0.90)[None, None])[0, 0] + 1e-6))
    g_all = np.array([lum_gain] * 3, np.float32) * 0.5 + gain * 0.5
    gmap = g_all[None, None] * (1 - cream[..., None]) + gain[None, None] * cream[..., None]
    rgb3 = np.clip(rgb2 * gmap, 0, 255)
    out = np.dstack([rgb3, a2 * 255.0]).round().clip(0, 255).astype(np.uint8)
    fixed = WORK / f'tail-sprite-fixed-{VER}.png'
    cv2.imwrite(str(fixed), out)
    print('  tail: haunch white', hw.round(1), 'cream before', (cur * 0.9).round(1), 'gain', gain.round(3), flush=True)
    return prep_tail_padded(fixed)


def deform_poly_i(rest, ph):
    poly = rest.copy()
    arc, total = Hh._arc(rest)
    s = arc / (total + 1e-8)
    tg = np.zeros_like(poly)
    tg[1:-1] = poly[2:] - poly[:-2]
    tg[0] = poly[1] - poly[0]
    tg[-1] = poly[-1] - poly[-2]
    tg = tg / (np.sqrt((tg ** 2).sum(1, keepdims=True)) + 1e-8)
    nrm = np.stack([-tg[:, 1], tg[:, 0]], 1)
    plant_dir = (Hh.TIP_W - Hh.BASE_W).astype(np.float32)
    plant_dir /= (np.hypot(*plant_dir) + 1e-8)
    two = np.sin(2 * np.pi * 2 * ph / NCYCLE)
    five = np.sin(2 * np.pi * 5 * ph / NCYCLE + 0.4)
    mid = 5.0 * two
    tip = 10.0 * two + 3.0 * five
    amp = np.where(s < 0.55, mid * np.clip(s / 0.55, 0, 1), mid + (tip - mid) * sstep((s - 0.55) / 0.45))
    curl = 0.0
    if 90.0 < ph < 130.0:
        u = (ph - 90.0) / 40.0
        curl = (0.5 - 0.5 * np.cos(2 * np.pi * u)) * CFG['tail_curl']
    open_ = CFG['tail_open'] * (s ** 1.8)
    lift = CFG['tail_lift'] * (s ** 2.0)
    off = nrm * (amp + lift)[:, None] + plant_dir[None, :] * (open_ + curl * s ** 1.7)[:, None]
    poly = poly + off
    poly[0] = rest[0]
    return poly


# ---------------- process ----------------
def clean_matte(m):
    """Drop small rembg false-positive islands (one sat on the glass at
    ~x444,y373 and kept a bright day-sky chip ungraded)."""
    a = m.astype(np.float32)
    sc = 255.0 if a.max() > 1.5 else 1.0
    bw = (a > 0.08 * sc).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(bw, 8)
    if n <= 2:
        return m
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= 4000
    mask = keep[lab]
    mask = cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    return np.where(mask, m, 0).astype(m.dtype)


def process(src, mattes, ctx, preview=None):
    Title = C.title_mask(); Steam = C.steam_mask(); Panes = ctx['panes']
    Lamp = R.lamp_masks(); Pool = R.lamp_pool()
    want = set(preview) if preview else set(range(NFR))
    out = [None] * NFR
    for i in range(NFR):
        if i not in want:
            continue
        ph = phase_of(i)
        w = 0.5 - 0.5 * np.cos(2.0 * np.pi * ph / NCYCLE)
        frame = src[i]
        cat = clean_matte(mattes[i])
        cat_p = R.cat_protect(cat)
        cat_a = cat.astype(np.float32)
        if cat_a.max() > 1.5:
            cat_a /= 255.0
        leaves = ctx['leaves']
        fixture = lamp_fixture_i(frame, Lamp, Panes)
        img, glass = grade_window_i(frame, ctx['bg'], ctx['bgv'], Panes, cat_p, fixture, leaves, Steam, Title, w, cat_a)
        img = img * (1.0 - fixture)[..., None] + frame.astype(np.float32) * fixture[..., None]
        img = R.grade_room(img, Pool, Lamp, cat_a, Title, ph, glass=glass)
        img = room_extra(img, cat_a, glass, ph)
        img = R.apply_grain(img, ph, cat_a, Title)
        img = np.clip(img, 0, 255).astype(np.uint8)
        img = apply_dust_i(img, ctx['motes'], ctx['unsafe_soft'], ph)
        img = G.apply_plant(img, ctx['plant_plate'], cat_a, ph)
        img, tip = Hh.apply_painted_tail(img, cat_a, ph, ctx['tail'], ctx['mic'])
        img = img.copy()
        y0, y1, x0, x1 = TITLE
        img[y0:y1, x0:x1] = frame[y0:y1, x0:x1]
        out[i] = img
        if (preview and len(want) < 30) or i % 50 == 0:
            Yi = luma(img)
            print(f'  n{i} ph{ph} glob={Yi.mean():.1f} face={Yi[300:340,700:740].mean():.1f} '
                  f'lamp={Yi[120:170,280:360].mean():.0f} tip=({tip[0]:.0f},{tip[1]:.0f})', flush=True)
    return out


_G = {}


def _work(chunk):
    o = process(_G['src'], _G['mattes'], _G['ctx'], preview=chunk)
    return [(i, o[i]) for i in chunk]


def process_parallel(src, mattes, ctx, nproc=6):
    import multiprocessing as mp
    _G.update(src=src, mattes=mattes, ctx=ctx)
    chunks = [list(range(k, NFR, nproc * 4)) for k in range(nproc * 4)]
    out = [None] * NFR
    with mp.get_context('fork').Pool(nproc) as pool:
        for res in pool.imap_unordered(_work, chunks):
            for i, f in res:
                out[i] = f
    assert all(o is not None for o in out)
    return out


def setup(src, mattes):
    Hh._deform_poly = deform_poly_i  # apply_painted_tail looks it up at call time
    a0 = mattes[0].astype(np.float32)
    a0 = a0 / 255.0 if a0.max() > 1.5 else a0
    ctx = {}
    print('window background…', flush=True)
    ctx['bg'], ctx['bgv'] = window_background(src, mattes)
    cv2.imwrite(str(QC / 'preview' / 'window-bg.png'), np.clip(ctx['bg'], 0, 255).astype(np.uint8)[80:410, 250:850])
    ctx['panes'], ctx['leaves'] = static_masks(ctx['bg'])
    _FIX['bg'] = ctx['bg']
    lamp_fixture_i(src[0], R.lamp_masks(), ctx['panes'])
    ctx['panes'] = np.clip(np.maximum(ctx['panes'], _FIX['ring']), 0, 1)
    _FIX.pop('shade'); _FIX.pop('ring')  # rebuild with final panes
    cv2.imwrite(str(QC / 'preview' / 'mask-panes-leaves.png'), (np.dstack([ctx['leaves'], ctx['panes'], ctx['panes']]) * 255).astype(np.uint8)[0:420, 250:870])
    ctx['plant_plate'] = G.build_plant_plate(src[0], a0)
    ctx['mic'] = Hh._mic_mask()
    motes, unsafe = build_motes(mattes)
    print(f'  motes kept {len(motes)} of {len(G.MOTES)}', flush=True)
    ctx['motes'] = motes
    ctx['unsafe_soft'] = cv2.GaussianBlur(unsafe, (0, 0), 6.0) * 0  # orbits already safe
    # graded reference for tail colour: grade frame 0 without the tail
    tmp = {'bg': ctx['bg'], 'bgv': ctx['bgv'], 'motes': [], 'unsafe_soft': ctx['unsafe_soft'],
           'plant_plate': ctx['plant_plate'], 'mic': ctx['mic'], 'tail': None}
    Title = C.title_mask(); Panes = ctx['panes']; Lamp = R.lamp_masks(); Pool = R.lamp_pool()
    f = src[0]
    ca = a0
    leaves = ctx['leaves']
    fixture = lamp_fixture_i(f, Lamp, Panes)
    img, glass = grade_window_i(f, ctx['bg'], ctx['bgv'], Panes, R.cat_protect(mattes[0]), fixture, leaves, C.steam_mask(), Title, 0.5)
    img = R.grade_room(img, Pool, Lamp, ca, Title, phase_of(0), glass=glass)
    img = room_extra(img, ca, glass, phase_of(0))
    ctx['tail'] = prep_tail(np.clip(img, 0, 255))
    return ctx


def crops(out, idx, dest):
    cr = {
        'tail': (480, 720, 920, 1280, 3), 'sky': (85, 205, 260, 840, 2), 'window': (80, 410, 250, 850, 2),
        'face': (200, 520, 520, 900, 2), 'mouth': (360, 480, 640, 780, 4), 'lamp': (40, 280, 200, 500, 2),
        'plant': (430, 720, 1000, 1280, 2), 'earL': (170, 260, 540, 650, 4), 'earR': (170, 270, 840, 960, 4),
        'desk': (300, 620, 100, 700, 2),
    }
    for i in idx:
        f = out[i]
        cv2.imwrite(str(dest / f'full-n{i}.png'), f)
        for tag, (y0, y1, x0, x1, s) in cr.items():
            c = cv2.resize(f[y0:y1, x0:x1], None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(str(dest / f'{tag}-n{i}.png'), c)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--preview', action='store_true')
    ap.add_argument('--frames', default='0,88,89,138,199,241,370,432')
    args = ap.parse_args()
    if Hh.sha256(V41H_MP4) != V41H_SHA:
        raise SystemExit('v41h changed')
    for label, path, expect in (('v41', Hh.V41_MP4, Hh.V41_SHA), ('v41d', Hh.V41D_MP4, Hh.V41D_SHA), ('pcm', Hh.PCM, Hh.PCM_SHA)):
        if Hh.sha256(path) != expect:
            raise SystemExit(f'{label} sha changed')
    print('frozen shas ok', flush=True)
    src = Hh.load_span(Hh.SRC)
    mattes = Hh.load_span(Hh.MATTE, gray=True)
    ctx = setup(src, mattes)
    if args.preview:
        idx = [int(x) for x in args.frames.split(',')]
        out = process(src, mattes, ctx, preview=idx)
        crops(out, idx, QC / 'preview')
        return
    out = process_parallel(src, mattes, ctx)
    y0, y1, x0, x1 = TITLE
    bad = [i for i, (a, b) in enumerate(zip(src, out)) if not np.array_equal(a[y0:y1, x0:x1], b[y0:y1, x0:x1])]
    print('title mismatches', len(bad) if bad else 'none (byte-identical)')
    los = WORK / f'{VER}-span-890-915-lossless.mkv'
    mp4 = P / f'{VER}-test-25s.mp4'
    Hh.write_lossless(out, los)
    Hh.encode_mp4(out, mp4)
    print(VER, 'mp4', Hh.sha256(mp4))
    print(VER, 'lossless', Hh.sha256(los))
    print('v41i still', Hh.sha256(V41H_MP4)); print('v41h still', Hh.sha256(P / 'v41h-test-25s.mp4'))


if __name__ == '__main__':
    main()
