#!/usr/bin/env python3
"""v41k: cyan cleanup, cartoon laptop, warm dust, lamp bloom, tail root,
per-leaf plant, and a conservative rigid head-turn follow.

Built from frozen v41d lossless. Own builder and own outputs.
Does not edit or import build_v41i.py / build_v41j.py / tail-sprite-fixed.png.
Does not overwrite earlier named encodes.
Audio PCM is the approved v41 stem (mux only).
"""
from __future__ import annotations
import hashlib, shutil, subprocess as sp, sys
from pathlib import Path
import cv2
import numpy as np

P = Path('/workspace/lofi-v41-meow')
sys.path.insert(0, str(P))
import build_v41c as C
import build_v41d as D
import build_v41f_r2 as R
import build_v41g as G
import build_v41h as Hh

VER = 'v41k'
QC = P / 'qc' / VER
for d in ('preview', 'encoded'):
    (QC / d).mkdir(parents=True, exist_ok=True)
WORK = P / VER
WORK.mkdir(exist_ok=True)
(WORK / 'executor-encode').mkdir(exist_ok=True)
(WORK / 'cartoons').mkdir(exist_ok=True)

SRC = P / 'v41d' / 'v41d-span-890-915-lossless.mkv'
PCM = P / 'work' / 'test-audio-890-915-v41.f32le'
MATTE = P / 'v41f' / 'cat-alpha-890-915.mkv'
SPRITE = P / 'v41j' / 'tail-sprite-v41j.png'
SPRITE_FALLBACK = P / 'v41h' / 'tail-sprite.png'
CARTOON_DIR = WORK / 'cartoons'

NFR, W, H, FPS, NCYCLE = 600, 1280, 720, 24, 241
TITLE = Hh.TITLE
Y, X = R.Y, R.X
sstep, luma, hex_bgr, phase_of = Hh.sstep, Hh.luma, Hh.hex_bgr, Hh.phase_of

# frozen encodes we must not change
V41_MP4, V41_SHA = P / 'v41-test-25s.mp4', 'f53cb1127c75775b740b555dd856e0002c784f80a2a3097118d8e197a794e95f'
V41D_MP4, V41D_SHA = P / 'v41d-test-25s.mp4', 'd1dab25187f4756969344036f50cd0ef89fac7c414d80006b8853acf04be2b34'
V41H_MP4, V41H_SHA = P / 'v41h-test-25s.mp4', 'c4d772598b5b9d98e576cd8c36e2e627bbc7a24c279aa994a5bfd8b752986bb0'
V41I_EXEC, V41I_SHA = P / 'v41i' / 'executor-encode' / 'v41i-test-25s.mp4', '204d0e863ceb04010b5abf701563af16ad537c460d114d1e6151d08f97aba4d1'
V41J_EXEC, V41J_SHA = P / 'v41j' / 'executor-encode' / 'v41j-test-25s.mp4', 'b344110d08088595874da9f4894aed3e4dbd627b653ace5edb49c48c97f72be8'
PCM_SHA = 'a635074f164c14fb3faabbf33c258703e9b7b9645eddbf8028f63a29573745a0'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for ch in iter(lambda: f.read(1 << 20), b''):
            h.update(ch)
    return h.hexdigest()


def load_span(path, gray=False):
    return R.load_span(path, gray=gray) if hasattr(R, 'load_span') else Hh.load_span(path, gray=gray)


# ---------- panes, hood, plate ----------
# Right pane extended to the wooden post so leftover day-sky on x820-848 is graded.
PANE_LEFT = np.array([[266, 0], [560, 0], [558, 394], [268, 396]], np.int32)
PANE_RIGHT = np.array([[600, 0], [848, 0], [846, 396], [600, 396]], np.int32)
HOOD_POLY = np.array([
    [215, 145], [215, 130], [225, 85], [250, 52], [280, 40], [320, 36],
    [355, 44], [380, 58], [405, 85], [422, 115], [432, 145], [422, 172],
    [400, 190], [360, 202], [320, 206], [280, 202], [245, 190], [225, 165],
], np.int32)


def pane_mask():
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [PANE_LEFT, PANE_RIGHT], 255)
    return cv2.GaussianBlur(m.astype(np.float32) / 255.0, (0, 0), 2.0)


def hood_region():
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [HOOD_POLY], 255)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.dilate(m, ker)
    return cv2.GaussianBlur(m.astype(np.float32) / 255.0, (0, 0), 1.0)


def lamp_fixture(frame):
    Lamp = R.lamp_masks()
    img = frame.astype(np.float32)
    B, Gch, Rc = img[..., 0], img[..., 1], img[..., 2]
    Yv = luma(img)
    metal = (
        sstep((Rc - B - 10.0) / 16.0)
        * sstep((Rc - Gch - 2.0) / 10.0)
        * sstep((Yv - 14.0) / 10.0)
        * sstep((135.0 - Yv) / 20.0)
    )
    hood = hood_region()
    fixture = np.clip(np.maximum(Lamp['fixture'], metal * hood), 0, 1)
    return cv2.GaussianBlur(np.clip(fixture, 0, 1), (0, 0), 0.8)


def indoor_leaves(frame, panes):
    leaf = C.green_leaf(frame)
    img = frame.astype(np.float32)
    B, Gch, Rc = img[..., 0], img[..., 1], img[..., 2]
    # Window dirt is teal (B >= G). Real leaves are green (G > B).
    leaf = leaf * sstep((Gch - B - 2.0) / 8.0)
    right = leaf * sstep((X - 790.0) / 20.0) * sstep((300.0 - Y) / 25.0)
    left = leaf * sstep((295.0 - X) / 18.0)
    desk = leaf * sstep((Y - 250.0) / 25.0) * sstep((520.0 - X) / 30.0) * sstep((X - 240.0) / 16.0)
    a = np.clip(np.maximum.reduce([right, left, desk]), 0, 1)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dil = cv2.dilate((a > 0.28).astype(np.uint8), ker).astype(np.float32)
    a = np.clip(a * 0.75 + dil * 0.35, 0, 1)
    a = a * sstep((panes - 0.02) / 0.12)
    return cv2.GaussianBlur(np.clip(a, 0, 1), (0, 0), 0.8)


def dusk_plate(w_breath):
    """Continuous ramp from y=0. Soft rose-gold lip, no horizontal step."""
    t = np.clip(Y / 396.0, 0, 1)
    upper = hex_bgr('0C1238')
    midn = hex_bgr('1A1F4A')
    blue = hex_bgr('2A3558')
    blue2 = hex_bgr('3A4868')
    blue3 = hex_bgr('4A5878')
    rose = hex_bgr('8A4A38') * (0.48 + 0.10 * w_breath)
    gold = hex_bgr('D4A565') * (0.26 + 0.08 * w_breath)
    stops = np.array([0.00, 0.08, 0.16, 0.24, 0.30, 0.36, 0.45, 0.58, 0.75, 1.00], np.float32)
    cols = np.stack([
        upper,
        upper * 0.55 + midn * 0.45,
        midn,
        midn * 0.40 + blue * 0.60,
        blue * 0.40 + blue2 * 0.32 + rose * 0.28,
        blue2 * 0.38 + rose * 0.32 + gold * 0.22 + blue3 * 0.08,
        blue2 * 0.55 + blue3 * 0.35 + gold * 0.10,
        blue3,
        blue3 * 0.75 + hex_bgr('5A6888') * 0.25,
        hex_bgr('5A6888'),
    ], 0)
    c = np.zeros((H, W, 3), np.float32)
    for k in range(3):
        c[..., k] = np.interp(t, stops, cols[:, k])
    return cv2.GaussianBlur(c, (0, 0), 10.0), t


# ---------- snow: isolated round flakes on the SOURCE frame ----------
def detect_and_paint_flakes(src, glass, land_a):
    Yf = luma(src)
    med = cv2.medianBlur(Yf.astype(np.uint8), 17).astype(np.float32)
    res = Yf - med
    res = cv2.GaussianBlur(res, (0, 0), 0.8)
    mu = cv2.GaussianBlur(Yf, (0, 0), 3.5)
    var = cv2.GaussianBlur((Yf - mu) ** 2, (0, 0), 3.5)
    tex = np.sqrt(np.clip(var, 0, None))
    textured = sstep((tex - 4.5) / 6.0)
    # Only count isolated peaks in relatively smooth sky; trees need a much
    # higher residual so clumps of snow on branches are not re-dotted.
    thr = 22.0 + 55.0 * textured
    cand = (res > thr) & (glass > 0.25) & (textured < 0.85)
    # local maxima in 5x5, isotropy via moments in 7x7
    ker = np.ones((5, 5), np.uint8)
    dil = cv2.dilate(res, ker)
    peaks = cand & (res >= dil - 1e-4)
    ys, xs = np.where(peaks)
    snow = np.zeros((H, W), np.float32)
    for y, x in zip(ys, xs):
        y0, y1 = max(0, y - 3), min(H, y + 4)
        x0, x1 = max(0, x - 3), min(W, x + 4)
        patch = res[y0:y1, x0:x1]
        mask = patch > (0.45 * res[y, x])
        if mask.sum() < 2:
            continue
        yy, xx = np.where(mask)
        # axis ratio from covariance
        if yy.size < 3:
            ar = 1.0
        else:
            cy, cx = yy.mean(), xx.mean()
            cov = np.cov(np.stack([yy - cy, xx - cx]))
            wsg, _ = np.linalg.eigh(cov)
            wsg = np.clip(wsg, 1e-4, None)
            ar = float(np.sqrt(wsg[1] / wsg[0]))
        if ar > 2.2:
            continue
        rad = float(np.clip(1.4 + 0.22 * min(mask.sum(), 12), 1.4, 3.4))
        strength = float(np.clip(10.0 + 0.35 * res[y, x], 10, 28))
        yy2, xx2 = np.mgrid[y - 6:y + 7, x - 6:x + 7]
        if yy2.min() < 0 or xx2.min() < 0 or yy2.max() >= H or xx2.max() >= W:
            continue
        g = np.exp(-0.5 * ((yy2 - y) ** 2 + (xx2 - x) ** 2) / (rad * rad))
        snow[y - 6:y + 7, x - 6:x + 7] = np.maximum(
            snow[y - 6:y + 7, x - 6:x + 7], g * strength)
    flake = np.array([1.06, 1.00, 0.92], np.float32)  # slightly cool white
    amt = 1.0 - 0.35 * land_a
    return snow[..., None] * flake * amt[..., None], snow


# ---------- chamfer nearest-fill (no scipy) ----------
def nearest_fill(color, sure, iters=16):
    """Propagate sure-pixel colours into the surrounding fringe. No blur."""
    h, w = sure.shape
    out = color.copy()
    known = sure.astype(np.uint8)
    # kill wrap at borders by leaving a 1px unknown frame
    known[0, :] = 0
    known[-1, :] = 0
    known[:, 0] = 0
    known[:, -1] = 0
    nbr = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))
    for _ in range(iters):
        if known.all():
            break
        acc = np.zeros_like(out)
        wsum = np.zeros((h, w), np.float32)
        for dy, dx in nbr:
            k = np.roll(np.roll(known, dy, 0), dx, 1)
            c = np.roll(np.roll(out, dy, 0), dx, 1)
            acc += c * k[..., None]
            wsum += k
        grow = (known == 0) & (wsum > 0)
        if not grow.any():
            break
        out[grow] = acc[grow] / wsum[grow, None]
        known[grow] = 1
    return out


def decontam_fringe(img, solid, dusk, glass, darken=0.92, erode_px=1):
    """Replace leftover day-sky only in the matte fringe. No dusk stroke into the glass."""
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode_px + 1, 2 * erode_px + 1))
    solid_u8 = (solid > 0.55).astype(np.uint8)
    sure = cv2.erode(solid_u8, ker)
    dil = cv2.dilate(solid_u8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    fringe = (dil > 0) & (sure == 0)
    if not fringe.any():
        return img
    B, Gch, Rc = img[..., 0], img[..., 1], img[..., 2]
    Yv = luma(img)
    cyan = ((B - Rc) > 8.0) & ((Gch - Rc) > 0.0) & (Yv > 32.0) & ((B + Gch) > 1.35 * Rc)
    hit = fringe & cyan
    if not hit.any():
        return img
    filled = nearest_fill(img, sure.astype(bool), iters=14)
    # Pixels still on the solid: pull interior colour. Pixels on the glass
    # side of the fringe: use the new sky. No wide dilation.
    on_solid = solid > 0.35
    mix = np.where(on_solid[..., None], filled * darken, dusk)
    out = img.copy()
    m = hit.astype(np.float32)[..., None]
    out = out * (1.0 - m) + mix * m
    return out


# ---------- window grade ----------
def grade_window(frame, panes, cat_p, fixture, leaves, steam, title, w_breath):
    # Choke cat and leaves so dusk sits under the fringe (eroded exclusion).
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cat_solid = cv2.erode((cat_p > 0.55).astype(np.uint8), ker).astype(np.float32)
    cat_choke = cv2.GaussianBlur(cat_solid, (0, 0), 0.6)
    leaf_solid = cv2.erode((leaves > 0.50).astype(np.uint8), ker).astype(np.float32)
    leaf_choke = cv2.GaussianBlur(leaf_solid, (0, 0), 0.6)
    glass = panes * (1.0 - cat_choke) * (1.0 - fixture) * (1.0 - leaf_choke) * (1.0 - 0.7 * steam) * (1.0 - title)
    glass = np.clip(glass, 0, 1)
    img = frame.astype(np.float32)
    if float(glass.max()) < 1e-4:
        return img, glass, None
    dusk, t = dusk_plate(w_breath)
    low = cv2.GaussianBlur(img, (0, 0), 4.0)
    Ylow = luma(low)
    Yf = luma(img)
    mu = cv2.GaussianBlur(Yf, (0, 0), 3.5)
    var = cv2.GaussianBlur((Yf - mu) ** 2, (0, 0), 3.5)
    tex = np.sqrt(np.clip(var, 0, None))
    textured = sstep((tex - 4.5) / 6.0)
    B, Gch, Rc = low[..., 0], low[..., 1], low[..., 2]
    teal = sstep(((B - Rc) - 16.0) / 28.0) * sstep(((B - Gch) - 0.0) / 18.0)
    sky = np.clip((1.0 - 0.92 * textured) * (0.55 + 0.45 * teal), 0, 1)
    sky = cv2.GaussianBlur(sky, (0, 0), 3.5)
    sky = np.clip(sky * glass, 0, 1)
    is_snow = sstep((Ylow - 130.0) / 40.0)
    Y_tree = 11.0 + 22.0 * t + (Ylow - 120.0) * 0.06
    Y_snow = 20.0 + 32.0 * t + (Ylow - 150.0) * 0.08
    Y2 = np.clip(Y_tree * (1.0 - is_snow) + Y_snow * is_snow, 7, 58)
    scale = np.clip(Y2 / (Ylow + 1e-3), 0.06, 0.40)
    cool = np.array([1.12, 0.98, 0.78], np.float32)
    land = low * scale[..., None] * cool
    ag = sstep((t - 0.22) / 0.10) * sstep((0.55 - t) / 0.18)
    ag = cv2.GaussianBlur(ag, (0, 0), 12.0)
    bounce = (0.044 + 0.022 * w_breath) * ag * np.clip(Ylow / 140.0, 0.1, 1.0)
    land = land + hex_bgr('C47A6A') * bounce[..., None] + hex_bgr('D4A565') * (0.28 * bounce)[..., None]
    land_a = np.clip((1.0 - sky) * glass, 0, 1)
    new_low = dusk * (1.0 - land_a)[..., None] + land * land_a[..., None]
    flakes, flake_m = detect_and_paint_flakes(img, glass, land_a)
    new_pane = np.clip(new_low + flakes, 0, 255)
    out = img * (1.0 - glass[..., None]) + new_pane * glass[..., None]
    return out, glass, (dusk, flake_m)


def lamp_flicker(ph):
    # Envelope used by the room grade: small, 241-periodic.
    return 0.85 + 0.15 * (0.55 + 0.45 * np.sin(2 * np.pi * 3 * ph / NCYCLE)
                          + 0.15 * np.sin(2 * np.pi * 7 * ph / NCYCLE + 0.6))


def kill_lamp_crescent(img, dusk, fixture, glass, ph):
    """Remove pale cyan-white crescent under the shade; optional warm bloom flickers."""
    hood = hood_region()
    # Protect the copper hood + warm cream interior, slightly dilated so we
    # don't nick the metal rim.
    B, Gch, Rc = img[..., 0], img[..., 1], img[..., 2]
    Yv = luma(img)
    warm_core = sstep((Rc - B - 8.0) / 14.0) * sstep((Yv - 40.0) / 30.0)
    protect = np.clip(sstep((hood - 0.18) / 0.12) + 0.85 * warm_core * hood, 0, 1)
    # Crescent zone under the right / lower lip.
    zone = sstep((X - 280.0) / 8.0) * sstep((450.0 - X) / 8.0) * sstep((Y - 95.0) / 8.0) * sstep((205.0 - Y) / 8.0)
    pale = sstep((Yv - 55.0) / 25.0) * sstep(((B + Gch) - 1.55 * Rc + 8.0) / 18.0)
    # also catch milky almost-white (high luma, weak chroma)
    chroma = np.maximum(np.maximum(Rc, Gch), B) - np.minimum(np.minimum(Rc, Gch), B)
    milky = sstep((Yv - 90.0) / 30.0) * sstep((28.0 - chroma) / 12.0)
    bad = np.clip(np.maximum(pale, milky) * zone * (1.0 - protect), 0, 1)
    # only on / next to glass, not inside the bowl
    near_glass = np.clip(glass + cv2.dilate((glass > 0.15).astype(np.uint8),
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))).astype(np.float32), 0, 1)
    bad = bad * sstep((near_glass - 0.08) / 0.15)
    bad = cv2.GaussianBlur(bad, (0, 0), 1.1)
    out = img * (1.0 - bad[..., None]) + dusk * bad[..., None]
    # Tight warm bloom on the glass under the lip, flickering with the lamp.
    fl = lamp_flicker(ph)
    ring = zone * (1.0 - protect) * sstep((near_glass - 0.1) / 0.2)
    ring = cv2.GaussianBlur(ring, (0, 0), 3.5)
    bloom = (0.07 * fl) * ring
    out = out + hex_bgr('D4A565') * (0.75 * bloom)[..., None] + hex_bgr('C47A6A') * (0.25 * bloom)[..., None]
    return np.clip(out, 0, 255)


def snowflake_mask(flake_m):
    if flake_m is None:
        return np.zeros((H, W), np.float32)
    return sstep((flake_m - 6.0) / 6.0)


def global_cyan_sweep(img, dusk, glass, fixture, flake_m, cat_a, leaves):
    """Replace leftover day-sky cyan at pane edges and mattes, except snowflakes.

    Do not touch already-graded trees or open sky — a wide sweep punched
    navy holes in the landscape.
    """
    B, Gch, Rc = img[..., 0], img[..., 1], img[..., 2]
    Yv = luma(img)
    cyan = (
        sstep((B - Rc - 16.0) / 8.0)
        * sstep((Gch - Rc - 4.0) / 10.0)
        * sstep((Yv - 36.0) / 14.0)
        * sstep(((B + Gch) - 1.55 * Rc) / 12.0)
    )
    # Restrict to: right/left/top pane-frame strips, and a tight matte fringe.
    edge = np.zeros((H, W), np.float32)
    # right post / pane edge
    edge += sstep((X - 816.0) / 3.0) * sstep((852.0 - X) / 3.0) * sstep((410.0 - Y) / 8.0)
    # left pane left edge
    edge += sstep((X - 258.0) / 3.0) * sstep((280.0 - X) / 3.0) * sstep((410.0 - Y) / 8.0)
    # top of both panes
    edge += sstep((8.0 - Y) / 5.0) * sstep((X - 260.0) / 6.0) * sstep((850.0 - X) / 6.0)
    # mullion sides
    edge += sstep((X - 548.0) / 3.0) * sstep((612.0 - X) / 3.0) * sstep((410.0 - Y) / 8.0)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    matte = np.maximum((cat_a > 0.25).astype(np.uint8), (leaves > 0.25).astype(np.uint8))
    fringe = cv2.dilate(matte, ker).astype(np.uint8) - cv2.erode(matte, ker)
    allow = np.clip(edge + fringe.astype(np.float32), 0, 1)
    snow = snowflake_mask(flake_m)
    hood = hood_region()
    solid = sstep((np.maximum(cat_a, leaves) - 0.70) / 0.12)
    bad = np.clip(cyan * allow * (1.0 - snow) * (1.0 - 0.90 * hood) * (1.0 - 0.80 * fixture) * (1.0 - solid), 0, 1)
    bad = sstep((bad - 0.22) / 0.20)
    if float(bad.max()) < 0.05:
        return img, bad
    out = img * (1.0 - bad[..., None]) + dusk * bad[..., None]
    return np.clip(out, 0, 255), bad


# ---------- cartoon laptop ----------
SHOT_PATHS = [
    CARTOON_DIR / 'shot-a-yarn-pounce.jpg',
    CARTOON_DIR / 'shot-b-yarn-chase.jpg',
    CARTOON_DIR / 'shot-c-bird-watch.jpg',
    CARTOON_DIR / 'shot-d-yarn-bat.jpg',
]
# integer partition of 241: 60+60+60+61
SHOT_LEN = [60, 60, 60, 61]
SHOT_START = [0, 60, 120, 180]
# spill colour sampled later; presets as warm/cool per shot (BGR)
SHOT_SPILL = [
    np.array([0x70, 0xC0, 0xE8], np.float32),  # sunny mint
    np.array([0x90, 0xB8, 0xF0], np.float32),  # peach/warm
    np.array([0xC8, 0xB0, 0x70], np.float32),  # sky blue-ish
    np.array([0x80, 0xC0, 0xF0], np.float32),  # cream/yellow
]


def shot_of(ph):
    p = int(ph) % NCYCLE
    if p < 60:
        return 0, p, 60
    if p < 120:
        return 1, p - 60, 60
    if p < 180:
        return 2, p - 120, 60
    return 3, p - 180, 61


def load_cartoons():
    shots = []
    for p in SHOT_PATHS:
        im = cv2.imread(str(p))
        if im is None:
            raise SystemExit(f'missing cartoon {p}')
        shots.append(im)
    return shots


def render_shot(im, local, length, sw, sh):
    """Scale up and pan/bounce 8–14 px over the shot."""
    # cover the unwarped screen with margin for a 14 px travel
    margin = 18
    tw, th = sw + 2 * margin, sh + 2 * margin
    canvas = cv2.resize(im, (tw, th), interpolation=cv2.INTER_AREA).astype(np.float32)
    u = local / max(length - 1, 1)
    # bounce + slight diagonal slide, 8–14 px in screen space
    dx = 10.0 * np.sin(2 * np.pi * u) + 4.0 * u
    dy = 8.0 * np.sin(2 * np.pi * u + 0.7) - 3.0 * np.cos(np.pi * u)
    # keep travel in 6–15 px range (amplitude already is)
    x0 = int(round(margin + dx))
    y0 = int(round(margin + dy))
    x0 = int(np.clip(x0, 0, tw - sw))
    y0 = int(np.clip(y0, 0, th - sh))
    return canvas[y0:y0 + sh, x0:x0 + sw]


def apply_cartoon(frame, shots, screen, ph):
    si, local, length = shot_of(ph)
    rect = render_shot(shots[si], local, length, D.SW, D.SH)
    back = cv2.warpPerspective(
        np.clip(rect, 0, 255).astype(np.uint8), D.H_INV, (W, H),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
    ).astype(np.float32)
    a = screen[..., None]
    out = frame.astype(np.float32) * (1.0 - a) + back * a
    # soft screen-light spill on desk / keyboard; colour + brightness change at cuts
    spill_col = SHOT_SPILL[si]
    # desk/keyboard under the laptop, not the bezel or paws
    spill = (sstep((X - 0.0) / 10.0) * sstep((250.0 - X) / 18.0)
             * sstep((Y - 548.0) / 12.0) * sstep((680.0 - Y) / 16.0))
    spill = cv2.GaussianBlur(np.clip(spill, 0, 1), (0, 0), 10.0)
    # slightly brighter just after a cut, then settles
    flash = 0.70 + 0.30 * np.exp(-0.5 * (local / 6.0) ** 2)
    # also a gentle in-shot pulse so it isn't static within a shot
    flash *= 0.92 + 0.08 * np.sin(2 * np.pi * local / max(length, 1))
    amt = 0.16 * flash * spill
    out = out * (1.0 - 0.35 * amt[..., None]) + spill_col * amt[..., None] + out * (0.20 * amt[..., None])
    return np.clip(out, 0, 255)


# ---------- warm, soft, out-of-focus dust ----------
def build_warm_motes():
    rng = np.random.RandomState(41)
    motes = []
    # fewer than v41j, larger, dimmer, lamp-cone bias
    for i in range(16):
        along = rng.uniform(50, 400)
        axis = np.array([0.20, 1.00], np.float32)
        axis = axis / np.hypot(*axis)
        hw = 16.0 + along * 0.48
        perp = rng.uniform(-0.80, 0.80) * hw
        pr = np.array([axis[1], -axis[0]])
        p = np.array([323.0, 148.0]) + along * axis + perp * pr
        motes.append(dict(
            x0=float(p[0]), y0=float(p[1]),
            rx=rng.uniform(6, 16), ry=rng.uniform(8, 22),
            jx=rng.uniform(1.0, 3.0), jy=rng.uniform(1.0, 3.0),
            k=int(rng.choice([1, 1, 2])),
            p1=rng.uniform(0, 2 * np.pi), p2=rng.uniform(0, 2 * np.pi),
            p3=rng.uniform(0, 2 * np.pi),
            rad=rng.uniform(4.5, 7.5),
            peak=rng.uniform(0.10, 0.20),
            mix=rng.uniform(0.0, 1.0),
            cone=True,
        ))
    for i in range(7):
        motes.append(dict(
            x0=rng.uniform(180, 520), y0=rng.uniform(260, 620),
            rx=rng.uniform(8, 20), ry=rng.uniform(6, 16),
            jx=rng.uniform(1.0, 2.5), jy=rng.uniform(1.0, 2.5),
            k=int(rng.choice([1, 2])),
            p1=rng.uniform(0, 2 * np.pi), p2=rng.uniform(0, 2 * np.pi),
            p3=rng.uniform(0, 2 * np.pi),
            rad=rng.uniform(5.0, 8.0),
            peak=rng.uniform(0.07, 0.14),
            mix=rng.uniform(0.0, 1.0),
            cone=False,
        ))
    return motes


WARM_A = np.array([0xC0, 0xE6, 0xFF], np.float32)  # #FFE6C0
WARM_B = np.array([0x90, 0xC8, 0xE8], np.float32)  # #E8C890
MOTES = build_warm_motes()


def apply_warm_dust(img, alpha, ph, face, panes):
    dust = np.zeros((H, W, 3), np.float32)
    two_pi = 2 * np.pi
    for m in MOTES:
        k = m['k']
        x = m['x0'] + m['rx'] * np.sin(two_pi * k * ph / NCYCLE + m['p1'])
        y = m['y0'] + m['ry'] * np.cos(two_pi * k * ph / NCYCLE + m['p2'])
        x += m['jx'] * np.sin(two_pi * (k + 3) * ph / NCYCLE + m['p3'])
        y += m['jy'] * np.cos(two_pi * (k + 2) * ph / NCYCLE + m['p1'])
        xi, yi = int(round(x)), int(round(y))
        rad = int(round(m['rad'] * 3.2))
        if xi - rad < 0 or yi - rad < 0 or xi + rad >= W or yi + rad >= H:
            continue
        if face[yi, xi] > 0.25 or panes[yi, xi] > 0.25:
            continue
        if yi < 600 and alpha[max(0, yi - 10):yi + 11, max(0, xi - 10):xi + 11].max() > 0.35:
            continue
        yy, xx = np.mgrid[yi - rad:yi + rad + 1, xi - rad:xi + rad + 1]
        # large soft disc, low peak, no hard core
        g = np.exp(-0.5 * ((yy - y) ** 2 + (xx - x) ** 2) / (m['rad'] * m['rad']))
        col = WARM_A * (1.0 - m['mix']) + WARM_B * m['mix']
        dust[yi - rad:yi + rad + 1, xi - rad:xi + rad + 1] += (g * m['peak'])[..., None] * col
    op = 0.42
    base = img.astype(np.float32)
    fur = cv2.dilate((alpha > 0.35).astype(np.uint8),
                     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))).astype(np.float32)
    fur = cv2.GaussianBlur(fur, (0, 0), 2.0) * sstep((620.0 - Y) / 24.0)
    prot = np.clip(np.maximum.reduce([face, fur, panes]), 0, 1)
    add = np.clip(dust * op * (1.0 - prot[..., None]), 0, 255)
    out = 255.0 - (255.0 - base) * (1.0 - add / 255.0)
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------- per-leaf plant reaction ----------
def right_plant_plate(frame, cat_a):
    """Green leaves in the right-corner plant, minus the cat and the pot."""
    leaf = C.green_leaf(frame)
    roi = sstep((X - 1000.0) / 16.0) * sstep((Y - 450.0) / 16.0)
    a = np.clip(leaf * roi * (1.0 - sstep((cat_a - 0.25) / 0.2)), 0, 1)
    # drop the dark pot / dirt at the very bottom
    a = a * sstep((700.0 - Y) / 14.0)
    return cv2.GaussianBlur(a, (0, 0), 1.0)


def segment_leaves(plate):
    """Connected components of the right-corner plant plate; attachment = nearest to STEM."""
    roi = np.zeros_like(plate, np.uint8)
    roi[450:H, 1000:W] = (plate[450:H, 1000:W] > 0.20).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(roi, connectivity=8)
    leaves = []
    stem = G.STEM
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < 80 or area > 18000:
            continue
        ys, xs = np.where(lab == i)
        d2 = (xs - stem[0]) ** 2 + (ys - stem[1]) ** 2
        k = int(np.argmin(d2))
        att = (float(xs[k]), float(ys[k]))
        mask = (lab == i).astype(np.float32)
        mask = cv2.GaussianBlur(mask, (0, 0), 0.8)
        # delay from distance to contact (near leaves first)
        delay = float(np.clip(np.sqrt(d2.min()) / 90.0, 0, 2.4))
        leaves.append(dict(mask=mask, att=att, delay=delay, area=area))
    leaves.sort(key=lambda L: L['delay'])
    return leaves


def apply_per_leaf(img, leaves, cat_a, ph):
    if not leaves:
        return img
    out = img.astype(np.float32)
    cat_d = cv2.GaussianBlur(
        cv2.dilate((cat_a > 0.30).astype(np.uint8),
                   cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))).astype(np.float32),
        (0, 0), 1.4)
    live_union = np.zeros((H, W), np.float32)
    warped = out.copy()
    wa_acc = np.zeros((H, W), np.float32)
    for L in leaves:
        ph_d = (ph - L['delay'] * FPS) % NCYCLE
        # wrap-safe: if delay would see a pre-impulse negative as late-cycle settle
        A = G.plant_amp(ph_d)
        if abs(A) < 0.04:
            continue
        ang = np.radians(4.2) * (A / G.PUSH_PX)
        ax, ay = L['att']
        vx = X - ax
        vy = Y - ay
        ca, sa = np.cos(ang), np.sin(ang)
        sx = ax + ca * vx + sa * vy
        sy = ay - sa * vx + ca * vy
        y0, y1, x0, x1 = 450, H, 1000, W
        mapx = X.copy()
        mapy = Y.copy()
        mapx[y0:y1, x0:x1] = sx[y0:y1, x0:x1]
        mapy[y0:y1, x0:x1] = sy[y0:y1, x0:x1]
        leaf_w = L['mask'] * (1.0 - cat_d)
        wr = cv2.remap(out, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        wa = cv2.remap(leaf_w, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        wa = np.clip(wa * (1.0 - cat_d), 0, 1)
        warped = warped * (1.0 - wa[..., None]) + wr * wa[..., None]
        wa_acc = np.clip(wa_acc + wa, 0, 1)
        live_union = np.maximum(live_union, leaf_w)
    if float(wa_acc.max()) < 0.04:
        return img
    vacated = np.clip(live_union * (1.0 - wa_acc) * (1.0 - cat_d), 0, 1)
    vmask = (vacated > 0.28).astype(np.uint8) * 255
    if vmask.any():
        filled = cv2.inpaint(np.clip(img, 0, 255).astype(np.uint8), vmask, 4, cv2.INPAINT_TELEA)
        vf = cv2.GaussianBlur(vacated, (0, 0), 1.0)
        base = img.astype(np.float32) * (1.0 - vf[..., None]) + filled.astype(np.float32) * vf[..., None]
    else:
        base = img.astype(np.float32)
    out = base * (1.0 - wa_acc[..., None]) + warped * wa_acc[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------- tail: pin the root, overlap fur, contact shadow ----------
# Shift the world base a little further under the haunch so it never translates
# into view. Tip target is unchanged.
TAIL_BASE = np.array([948.0, 642.0], np.float32)
TAIL_TIP = np.array([1170.0, 575.0], np.float32)


def prep_tail(sprite_path):
    # Reuse Hh._prep_tail mechanics by temporarily pointing its constants.
    old_base, old_tip, old_sprite = Hh.BASE_W, Hh.TIP_W, getattr(Hh, 'SPRITE', None)
    Hh.BASE_W = TAIL_BASE
    Hh.TIP_W = TAIL_TIP
    # Hh._prep_tail reads Hh.SPRITE if it uses a global; call the function that
    # loads from its own path if needed. We use the public prep if available.
    try:
        if hasattr(Hh, '_prep_tail'):
            # monkeypatch path if the function reads SPRITE
            if old_sprite is not None:
                Hh.SPRITE = Path(sprite_path)
            tail = Hh._prep_tail()
        else:
            raise AttributeError
    finally:
        Hh.BASE_W = old_base
        Hh.TIP_W = old_tip
        if old_sprite is not None:
            Hh.SPRITE = old_sprite
    return tail


def apply_tail_pinned(frame, cat_a, ph, tail, mic):
    """Same slice-deform as v41h, but the base is locked under the body with
    a stronger fur overlap and a soft contact shadow."""
    # Temporarily use our base/tip inside Hh._deform_poly / apply
    old_base, old_tip = Hh.BASE_W, Hh.TIP_W
    Hh.BASE_W = TAIL_BASE
    Hh.TIP_W = TAIL_TIP
    try:
        img, tip = Hh.apply_painted_tail(frame, cat_a, ph, tail, mic)
    finally:
        Hh.BASE_W = old_base
        Hh.TIP_W = old_tip
    img = img.astype(np.float32)
    # Fur overlap at the root: the cat body always wins in a soft radius
    # around the pinned base, so the sprite cannot read as a 2D card sliding up.
    cat = np.clip(cat_a, 0, 1)
    cat_d = cv2.GaussianBlur(
        cv2.dilate((cat > 0.35).astype(np.uint8),
                   cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))).astype(np.float32),
        (0, 0), 1.6)
    root = np.exp(-0.5 * ((X - TAIL_BASE[0]) ** 2 + (Y - TAIL_BASE[1]) ** 2) / (58.0 ** 2))
    # Underlap: put the original body back where the root meets fur.
    overlap = np.clip(cat_d * (0.35 + 0.80 * root), 0, 1)
    img = img * (1.0 - overlap[..., None]) + frame.astype(np.float32) * overlap[..., None]
    # Soft contact shadow just behind the root, on chair / leaves, not on fur.
    sh = cv2.GaussianBlur(root, (0, 0), 10.0)
    shadow = 0.28 * sh * (1.0 - cat) * sstep((Y - (TAIL_BASE[1] - 20)) / 16.0)
    img = img * (1.0 - shadow[..., None])
    return np.clip(img, 0, 255).astype(np.uint8), tip


# ---------- conservative rigid head follow during the 15.4s turn ----------
def follow_head_turn(img, src_before, cat_a, i):
    """Move the whole head (silhouette + features) together during the turn.

    Features already slide left/up; we add a small rigid rotation so the
    ears and crown travel with them. No texture stretch.
    """
    if i < 345 or i > 405:
        return img
    # envelope peaks around n370 (15.4 s)
    if i <= 370:
        wt = float(sstep((i - 345) / 18.0))
    else:
        wt = float(1.0 - sstep((i - 370) / 28.0))
    if wt < 0.04:
        return img
    # ~2.0 degrees at the peak, about a 12-15 px ear travel at the crown
    ang = np.radians(-2.05) * wt
    pivot = (690.0, 505.0)
    M = cv2.getRotationMatrix2D(pivot, np.degrees(ang), 1.0)
    head = sstep((520.0 - (Y - 80.0)) / 40.0) * sstep((X - 520.0) / 20.0) * sstep((960.0 - X) / 20.0)
    head = np.clip(head * sstep((cat_a - 0.15) / 0.25), 0, 1)
    head = cv2.GaussianBlur(head, (0, 0), 3.0)
    rot = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    rot_a = cv2.warpAffine(head, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    a = np.clip(rot_a, 0, 1)
    out = img.astype(np.float32) * (1.0 - a[..., None]) + rot.astype(np.float32) * a[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def scrub_pane_cyan(img, panes, dusk, leaves, cat_a, fixture, flake_m):
    """Inpaint leftover day-sky teal on the pane (window dirt, matte leaks)."""
    B, Gch, Rc = img[..., 0], img[..., 1], img[..., 2]
    Yv = luma(img)
    teal = ((B - Rc) > 12.0) & ((B + Gch) > 1.4 * Rc + 8.0) & (Yv > 34.0) & (B >= Gch - 2.0)
    near = cv2.dilate((panes > 0.12).astype(np.uint8),
                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    snow = snowflake_mask(flake_m)
    leaf_core = leaves > 0.45
    cat_core = cat_a > 0.45
    hood = hood_region()
    hit = teal & (near > 0) & (~leaf_core) & (~cat_core) & (hood < 0.4) & (fixture < 0.4) & (snow < 0.3)
    if not hit.any():
        return img
    mask = hit.astype(np.uint8) * 255
    # close pinholes so Telea doesn't leave crumbs
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    src = np.clip(img, 0, 255).astype(np.uint8)
    filled = cv2.inpaint(src, mask, 6, cv2.INPAINT_TELEA).astype(np.float32)
    m = (mask > 0).astype(np.float32)
    m = cv2.GaussianBlur(m, (0, 0), 0.7)
    out = img * (1.0 - m[..., None]) + filled * m[..., None]
    return np.clip(out, 0, 255)


def process(src, mattes, plant_leaves, face, panes_solid, tail, mic, shots, preview=None):
    Title = C.title_mask()
    Steam = C.steam_mask()
    Panes = pane_mask()
    Lamp = R.lamp_masks()
    Pool = R.lamp_pool()
    screen = D.screen_mask()
    want = set(preview) if preview else set(range(NFR))
    out = [None] * NFR
    for i in range(NFR):
        if i not in want:
            continue
        ph = phase_of(i)
        w = 0.5 - 0.5 * np.cos(2.0 * np.pi * ph / NCYCLE)
        frame = src[i]
        cat = mattes[i]
        cat_a = cat.astype(np.float32)
        if cat_a.max() > 1.5:
            cat_a = cat_a / 255.0
        cat_p = R.cat_protect(cat)
        leaves = indoor_leaves(frame, Panes)
        fixture = lamp_fixture(frame)
        img, glass, extra = grade_window(frame, Panes, cat_p, fixture, leaves, Steam, Title, w)
        dusk, flake_m = extra if extra is not None else (dusk_plate(w)[0], None)
        # restore the real fixture (source copper / cream) after the glass grade
        img = img * (1.0 - fixture[..., None]) + frame.astype(np.float32) * fixture[..., None]
        img = decontam_fringe(img, cat_p, dusk, glass, darken=0.90, erode_px=1)
        img = decontam_fringe(img, leaves, dusk, glass, darken=0.88, erode_px=1)
        img = kill_lamp_crescent(img, dusk, fixture, glass, ph)
        img, _cyan = global_cyan_sweep(img, dusk, glass, fixture, flake_m, cat_a, leaves)
        img = scrub_pane_cyan(img, Panes, dusk, leaves, cat_a, fixture, flake_m)
        img = R.grade_room(img, Pool, Lamp, cat_a, Title, ph, glass=glass)
        img = R.apply_grain(img, ph, cat_a, Title)
        img = np.clip(img, 0, 255).astype(np.uint8)
        # head follow after grade so vacated pixels keep the new sky
        img = follow_head_turn(img, frame, cat_a, i)
        img = apply_cartoon(img, shots, screen, ph)
        img = apply_warm_dust(img, cat_a, ph, face, panes_solid)
        img = apply_per_leaf(img, plant_leaves, cat_a, ph)
        img, tip = apply_tail_pinned(img, cat_a, ph, tail, mic)
        img = img.copy()
        y0, y1, x0, x1 = TITLE
        img[y0:y1, x0:x1] = frame[y0:y1, x0:x1]
        out[i] = img
        if preview or i % 25 == 0:
            Yi = luma(img)
            print(f'  n{i} ph{ph} glob={Yi.mean():.1f} tip=({tip[0]:.0f},{tip[1]:.0f}) '
                  f'shot={shot_of(ph)[0]} skyL={Yi[20, 400]:.0f} skyR={Yi[20, 720]:.0f}',
                  flush=True)
    return out


def contact_sheet(frames, labels, path):
    thumbs = []
    for f, lab in zip(frames, labels):
        t = cv2.resize(f, (640, 360), interpolation=cv2.INTER_AREA)
        cv2.putText(t, lab, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2, cv2.LINE_AA)
        thumbs.append(t)
    cols = 4
    rows = int(np.ceil(len(thumbs) / cols))
    while len(thumbs) < rows * cols:
        thumbs.append(np.zeros((360, 640, 3), np.uint8))
    grid = np.vstack([np.hstack(thumbs[r * cols:(r + 1) * cols]) for r in range(rows)])
    cv2.imwrite(str(path), grid, [int(cv2.IMWRITE_JPEG_QUALITY), 92])


def save_preview(out, idx, dest):
    dest.mkdir(parents=True, exist_ok=True)
    crops = {
        'head': (160, 420, 560, 920),
        'lamp': (90, 210, 270, 460),
        'plant': (250, 430, 260, 440),
        'sill': (340, 440, 280, 450),
        'rightedge': (0, 250, 800, 880),
        'screen': (360, 570, 0, 230),
        'tail': (430, 720, 820, 1280),
        'dust': (250, 500, 350, 700),
        'mouth': (280, 520, 560, 900),
        'sky': (0, 320, 280, 880),
    }
    sheet, labs = [], []
    for i in idx:
        if out[i] is None:
            continue
        cv2.imwrite(str(dest / f'full-n{i}.png'), out[i])
        for tag, (y0, y1, x0, x1) in crops.items():
            c = out[i][y0:y1, x0:x1]
            c3 = cv2.resize(c, (c.shape[1] * 3, c.shape[0] * 3), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(str(dest / f'{tag}-3x-n{i}.jpg'), c3, [int(cv2.IMWRITE_JPEG_QUALITY), 93])
        sky = out[i][0:320, 280:880]
        cv2.imwrite(str(dest / f'sky-y0-n{i}.png'), sky)
        sheet.append(out[i])
        labs.append(f'n{i} ph{phase_of(i)}')
    if sheet:
        contact_sheet(sheet, labs, dest / 'contact-sheet.jpg')
    print('preview written', dest)


def encode_mp4(frames, path):
    Hh.encode_mp4(frames, path) if hasattr(Hh, 'encode_mp4') else R.encode_mp4(frames, path)


def write_lossless(frames, path):
    if hasattr(Hh, 'write_lossless'):
        Hh.write_lossless(frames, path)
    else:
        R.write_lossless(frames, path)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--preview', action='store_true')
    args = ap.parse_args()

    for label, path, expect in (
        ('v41', V41_MP4, V41_SHA),
        ('v41d', V41D_MP4, V41D_SHA),
        ('v41h', V41H_MP4, V41H_SHA),
        ('v41i-exec', V41I_EXEC, V41I_SHA),
        ('v41j-exec', V41J_EXEC, V41J_SHA),
        ('pcm', PCM, PCM_SHA),
    ):
        got = sha256(path)
        if got != expect:
            raise SystemExit(f'{label} sha changed: {got}')
    print('frozen shas ok', flush=True)

    sprite = SPRITE if SPRITE.exists() else SPRITE_FALLBACK
    if not sprite.exists():
        raise SystemExit('missing tail sprite')
    print('using sprite', sprite, flush=True)
    shots = load_cartoons()
    print('loading v41d lossless…', flush=True)
    src = load_span(SRC)
    assert len(src) == NFR, len(src)
    print('loading cat mattes…', flush=True)
    mattes = load_span(MATTE, gray=True)
    assert len(mattes) == NFR, len(mattes)
    a0 = mattes[0].astype(np.float32)
    if a0.max() > 1.5:
        a0 = a0 / 255.0
    plant_plate = right_plant_plate(src[0], a0)
    if float(plant_plate.max()) < 0.2:
        plant_plate = G.build_plant_plate(src[0], a0)
    plant_leaves = segment_leaves(plant_plate)
    print(f'  segmented {len(plant_leaves)} right-plant leaves', flush=True)
    face = G.face_protect()
    panes_solid = G.pane_solid()
    print('preparing tail…', flush=True)
    # Point Hh.SPRITE at our copy for _prep_tail
    if not hasattr(Hh, 'SPRITE'):
        Hh.SPRITE = sprite
    old = Hh.SPRITE
    Hh.SPRITE = sprite
    Hh.BASE_W = TAIL_BASE
    Hh.TIP_W = TAIL_TIP
    tail = Hh._prep_tail()
    Hh.SPRITE = old
    mic = Hh._mic_mask()

    if args.preview:
        idx = [0, 30, 60, 88, 89, 90, 120, 138, 180, 199, 241, 369, 370, 432]
        out = process(src, mattes, plant_leaves, face, panes_solid, tail, mic, shots, preview=idx)
        save_preview(out, idx, QC / 'preview')
        return

    out = process(src, mattes, plant_leaves, face, panes_solid, tail, mic, shots)
    y0, y1, x0, x1 = TITLE
    bad = [i for i, (a, b) in enumerate(zip(src, out))
           if not np.array_equal(a[y0:y1, x0:x1], b[y0:y1, x0:x1])]
    print('title mismatches' if bad else 'title core byte-identical', len(bad) if bad else '', flush=True)
    los = WORK / 'v41k-span-890-915-lossless.mkv'
    mp4 = P / 'v41k-test-25s.mp4'
    print('writing lossless…', flush=True)
    write_lossless(out, los)
    print('encoding mp4…', flush=True)
    encode_mp4(out, mp4)
    print('v41k mp4', sha256(mp4), flush=True)
    print('v41k lossless', sha256(los), flush=True)
    # immediate backup in case a parallel run touches the public path
    enc = WORK / 'executor-encode'
    shutil.copy2(mp4, enc / 'v41k-test-25s.mp4')
    shutil.copy2(los, enc / 'v41k-span-890-915-lossless.mkv')
    shutil.copy2(Path(__file__), enc / 'build_v41k.py')
    for label, path in (('v41', V41_MP4), ('v41d', V41D_MP4), ('v41h', V41H_MP4),
                        ('v41i-exec', V41I_EXEC), ('v41j-exec', V41J_EXEC)):
        print(label, 'still', sha256(path), flush=True)


if __name__ == '__main__':
    main()
