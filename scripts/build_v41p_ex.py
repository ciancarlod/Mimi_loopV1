#!/usr/bin/env python3
"""v41p-ex: post-pass on frozen v41o-ex.

Keeps v41o-ex dusk sky (navy + purple/pink afterglow), darker room and
fuller tail. Restores window frost, snowy hill and falling flakes from
v41l (as visible, under the dusk grade). Applies v41l-style crown/ear
decontamination so there is no light or dark rim. Regrades the laptop
with v41l dim/keystone/glow treatment but the v41o-ex full-screen quad
and subject crop so the cartoon fills the bezel. Uses v41l warm dust.

Does not edit or overwrite earlier named encodes or builders.
Audio is the approved v41 PCM (mux only). Title is copied through.
241-periodic.
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
import build_v41c as C
import build_v41g as G
import build_v41h as Hh

VER = 'v41p-ex'
WORK = P / VER
QC = P / 'qc' / VER
SRC_O = P / 'v41o-ex' / 'v41o-ex-span-890-915-lossless.mkv'
SRC_L = P / 'v41l' / 'executor-encode' / 'v41l-span-890-915-lossless.mkv'
MATTE = Hh.MATTE
PCM = Hh.PCM

V41O_SHA = '5880689a465428c799e1d3c818363f6d8b5c88875668292401cf112a7827133e'
V41O_LOS_SHA = 'c08c418d6a15edbf0d6493187fe6a512980b536a09ee3ee725062758be081700'
V41L_LOS_SHA = 'dab09d13022d9a15c323b5ffb8de165d99d0c3521cfa766e65a118efb4f4128d'

NFR, W, H, NCYCLE = 600, 1280, 720, 241
TITLE = Hh.TITLE
Y, X = np.mgrid[0:H, 0:W].astype(np.float32)
phase_of = Hh.phase_of

CARTOON_DIR = P / 'v41k' / 'cartoons'
SHOT_PATHS = [CARTOON_DIR / n for n in (
    'shot-a-yarn-pounce.jpg', 'shot-b-yarn-chase.jpg',
    'shot-c-bird-watch.jpg', 'shot-d-yarn-bat.jpg',
)]
SHOT_START = [0, 60, 120, 180]
SHOT_LEN = [60, 60, 60, 61]
SHOT_CX = [600, 630, 627, 640]
CROP_W = 880

# Full visible lid opening from v41o-ex (fills the bezel, including the slanted bottom).
SCR_Q = np.array([[-2, 377], [161, 377], [193, 554], [-2, 590]], np.float32)
SW, SH = 400, 512
SCR_HINV = cv2.getPerspectiveTransform(
    np.array([[0, 0], [SW - 1, 0], [SW - 1, SH - 1], [0, SH - 1]], np.float32),
    SCR_Q,
)

PANE_L = np.array([[266, 0], [558, 0], [556, 400], [268, 400]], np.int32)
PANE_R = np.array([[600, 0], [840, 0], [838, 400], [600, 400]], np.int32)

HOOD_POLY = np.array([
    [215, 145], [215, 130], [225, 85], [250, 52], [280, 40], [320, 36],
    [355, 44], [380, 58], [405, 85], [422, 115], [432, 145], [422, 172],
    [400, 190], [360, 202], [320, 206], [280, 202], [245, 190], [225, 165],
], np.int32)


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
    return np.clip(x, 0, 1)


def pane_mask():
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [PANE_L, PANE_R], 1)
    return m


def hood_mask():
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [HOOD_POLY], 255)
    return m


def nearest_fill(color, sure, iters=16):
    h, w = sure.shape
    out = color.copy()
    known = sure.astype(np.uint8)
    known[0, :] = 0
    known[-1, :] = 0
    known[:, 0] = 0
    known[:, -1] = 0
    nbr = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))
    for _ in range(iters):
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


# ---------- snow: frost + hill from v41l, flakes from v41l tophat ----------
_PANES = None
_HOOD = None


def restore_snow(dst, donor, cat_a):
    global _PANES, _HOOD
    if _PANES is None:
        _PANES = pane_mask()
        _HOOD = hood_mask()
    a = np.clip(cat_a, 0, 1).astype(np.float32)
    ker5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    ker3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    ker7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    cat = cv2.dilate((a > 0.12).astype(np.uint8), ker5)
    lamp = cv2.dilate((_HOOD > 0).astype(np.uint8), ker7)
    open_g = (_PANES > 0) & (cat == 0) & (lamp == 0)
    Yd = luma(donor)
    B, Gch, Rc = donor[..., 0], donor[..., 1], donor[..., 2]
    chroma = np.maximum(np.maximum(Rc, Gch), B) - np.minimum(np.minimum(Rc, Gch), B)
    # Frosted snow / snowy hill along the bottom of the glass.
    # The banks include mid-grey and slightly warm lamp-lit snow. Only
    # clearly green indoor-leaf pixels are excluded.
    lower = (Y > 262) & (Y < 408)
    leaf = (Gch > Rc + 12.0) & (Gch > B + 8.0) & (Gch > 40.0)
    frost = open_g & lower & (Yd > 40) & (chroma < 90) & (~leaf)
    frost = frost.astype(np.uint8)
    frost = cv2.morphologyEx(frost, cv2.MORPH_OPEN, ker3)
    ker21 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    frost = cv2.morphologyEx(frost, cv2.MORPH_CLOSE, ker21)
    frost = frost & open_g.astype(np.uint8) & lower.astype(np.uint8) & (~leaf).astype(np.uint8)
    # Falling flakes: small compact highlights in the donor pane, not big dirt.
    th = cv2.morphologyEx(Yd, cv2.MORPH_TOPHAT, ker7)
    flakes = open_g & (th > 16) & (Yd > 38) & (chroma < 42) & (Y < 360)
    nlab, lab, stats, _ = cv2.connectedComponentsWithStats(flakes.astype(np.uint8), 8)
    keep = np.zeros_like(flakes)
    for i in range(1, nlab):
        area = int(stats[i, cv2.CC_STAT_AREA])
        bw, bh = int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])
        if 2 <= area <= 70 and bw <= 12 and bh <= 12:
            keep[lab == i] = 1
    flakes = keep
    if not frost.any() and not flakes.any():
        return dst
    out = dst.astype(np.float32)
    don = donor.astype(np.float32)
    # Slight dusk warmth so the frost sits in the late-sunset pane, still bright.
    frost_col = don * np.array([0.94, 0.98, 1.02], np.float32)
    if frost.any():
        m = cv2.GaussianBlur(frost.astype(np.float32), (0, 0), 1.8)
        m = np.clip(m, 0, 1)
        out = out * (1.0 - m[..., None]) + frost_col * m[..., None]
    if flakes.any():
        m = cv2.GaussianBlur(flakes.astype(np.float32), (0, 0), 0.55)
        m = np.clip(m * 0.92, 0, 1)
        flake_col = don * np.array([0.96, 0.99, 1.02], np.float32)
        out = out * (1.0 - m[..., None]) + flake_col * m[..., None]
    return np.clip(out, 0, 255)


# ---------- crown / ear: no light rim, no dark navy line ----------
def decontam_crown(img, cat_a):
    win = sstep((260.0 - Y) / 16.0) * sstep((X - 430.0) / 14.0) * sstep((900.0 - X) / 16.0)
    if float(win.max()) < 0.1:
        return img
    a = np.clip(cat_a, 0, 1).astype(np.float32)
    ker1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    ker2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    ker8 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    solid = (a > 0.50).astype(np.uint8)
    sure = cv2.erode(solid, ker1)
    peaks = solid & (cv2.dilate(sure, ker1) == 0)
    sure = np.clip(sure + peaks, 0, 1).astype(np.uint8)
    near = cv2.dilate(solid, ker8)
    if int(sure.sum()) < 40:
        return img
    fur = nearest_fill(img, sure.astype(bool), iters=18)
    sky_seed = (near == 0) & (win > 0.4) & (a < 0.08)
    if int(sky_seed.sum()) < 20:
        sky_seed = (win > 0.5) & (a < 0.05)
    sky = nearest_fill(img, sky_seed.astype(bool), iters=18)
    B, Gch, Rc = img[..., 0], img[..., 1], img[..., 2]
    Yv = luma(img)
    warm = Rc - B
    # Light/pale rim or a dark navy stroke along the silhouette, against the sky.
    pale = ((B - Rc) > 2.0) & (Yv > 22.0) & (Yv < 95.0) & (warm < 8.0) & (near > 0)
    dark = (Yv < 32.0) & (warm < 20.0) & (near > 0) & (sure == 0)
    edge = cv2.dilate(solid, ker2) & (cv2.erode(solid, ker2) == 0)
    cool_on_fur = (edge > 0) & (win > 0.15) & (warm < 12.0) & (Yv < 95.0)
    fur_edge = cv2.dilate(sure, ker2)
    sky_zone = ((near > 0) | pale | dark) & (fur_edge == 0) & (win > 0.15)
    over = ((fur_edge > 0) & (sure == 0) & (win > 0.15)) | (cool_on_fur & (fur_edge > 0))
    if not (sky_zone.any() or over.any()):
        return img
    out = img.copy().astype(np.float32)
    if sky_zone.any():
        m = sky_zone.astype(np.float32)
        m = np.clip(cv2.GaussianBlur(m, (0, 0), 0.55), 0, 1)
        m[sure > 0] = 0
        out = out * (1.0 - m[..., None]) + sky * m[..., None]
    if over.any():
        m = over.astype(np.float32)
        m = np.clip(cv2.GaussianBlur(m, (0, 0), 0.45), 0, 1)
        out = out * (1.0 - m[..., None]) + fur * m[..., None]
    return np.clip(out, 0, 255)


# ---------- laptop: v41l grade, v41o-ex full quad + subject crop ----------
_SHOTS = []
_SCRM = None


def scr_mask():
    global _SCRM
    if _SCRM is None:
        big = np.zeros((H * 4, W * 4), np.uint8)
        cv2.fillPoly(big, [np.round(SCR_Q * 4).astype(np.int32)], 255)
        m = cv2.resize(big, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        _SCRM = cv2.GaussianBlur(m, (0, 0), 0.8)
    return _SCRM


def load_cartoons():
    global _SHOTS
    if not _SHOTS:
        for k, p in enumerate(SHOT_PATHS):
            im = cv2.imread(str(p))
            if im is None:
                raise SystemExit(f'missing cartoon {p}')
            x0 = int(np.clip(SHOT_CX[k] - CROP_W // 2, 0, im.shape[1] - CROP_W))
            _SHOTS.append(im[:, x0:x0 + CROP_W])
    return _SHOTS


def shot_of(ph):
    p = int(ph) % NCYCLE
    for k in (3, 2, 1, 0):
        if p >= SHOT_START[k]:
            return k, p - SHOT_START[k], SHOT_LEN[k]


def render_shot(im, local, length, sw, sh):
    margin = 18
    tw, th = sw + 2 * margin, sh + 2 * margin
    canvas = cv2.resize(im, (tw, th), interpolation=cv2.INTER_AREA).astype(np.float32)
    u = local / float(length)
    dx = 9.0 * np.sin(2 * np.pi * u)
    dy = 6.0 * np.sin(2 * np.pi * u + 0.7)
    M = np.float32([[1, 0, -(margin + dx)], [0, 1, -(margin + dy)]])
    return cv2.warpAffine(canvas, M, (sw, sh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def grade_rect(rect):
    """Dim to the darker dusk room; peak well below the lamp bulb."""
    r = rect.astype(np.float32)
    r = r * 0.42
    r = r * np.array([0.90, 0.96, 1.02], np.float32)
    yy, xx = np.mgrid[:SH, :SW].astype(np.float32)
    cy, cx = (SH - 1) / 2.0, (SW - 1) / 2.0
    rho = np.sqrt(((yy - cy) / (SH * 0.62)) ** 2 + ((xx - cx) / (SW * 0.62)) ** 2)
    vig = 1.0 - 0.28 * np.clip(rho, 0, 1) ** 2
    r = r * vig[..., None]
    scan = 1.0 - 0.024 * (0.5 + 0.5 * np.sin(yy * np.pi / 2.2))
    r = r * scan[..., None]
    r = cv2.GaussianBlur(r, (0, 0), 0.50)
    bloom = cv2.GaussianBlur(r, (0, 0), 2.8)
    r = np.clip(r * 0.88 + bloom * 0.18, 0, 255)
    return r


def apply_screen(frame, shots, ph):
    si, local, length = shot_of(ph)
    rect = grade_rect(render_shot(shots[si], local, length, SW, SH))
    if local < 5:
        pk = (si - 1) % 4
        prev = grade_rect(render_shot(shots[pk], SHOT_LEN[pk] + local, SHOT_LEN[pk], SW, SH))
        t = (local + 1) / 6.0
        rect = prev * (1.0 - t) + rect * t
    pictured = cv2.warpPerspective(
        np.clip(rect, 0, 255).astype(np.uint8), SCR_HINV, (W, H),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
    ).astype(np.float32)
    a = scr_mask()
    img = frame.astype(np.float32)
    img = img * (1.0 - a[..., None]) + pictured * a[..., None]
    # thin bezel contact shadow + a little glow, no inset mat
    edge = cv2.dilate(a, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    edge = np.clip(edge - cv2.erode(a, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))), 0, 1)
    edge = cv2.GaussianBlur(edge, (0, 0), 1.0)
    img = img * (1.0 - 0.28 * edge[..., None])
    glow = cv2.GaussianBlur(a, (0, 0), 4.0)
    glow = np.clip(glow - a, 0, 1)
    mean_c = pictured.reshape(-1, 3).mean(axis=0) * 0.30
    img = img + glow[..., None] * mean_c * 0.50
    return np.clip(img, 0, 255)


# ---------- v41l warm dust ----------
WARM_A = np.array([0xC0, 0xE6, 0xFF], np.float32)
WARM_B = np.array([0x90, 0xC8, 0xE8], np.float32)


def build_motes():
    rng = np.random.RandomState(41)
    motes = []
    for i in range(20):
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
            rad=rng.uniform(4.8, 8.0),
            peak=rng.uniform(0.20, 0.34),
            mix=rng.uniform(0.0, 1.0),
        ))
    for i in range(8):
        motes.append(dict(
            x0=rng.uniform(180, 520), y0=rng.uniform(260, 620),
            rx=rng.uniform(8, 20), ry=rng.uniform(6, 16),
            jx=rng.uniform(1.0, 2.5), jy=rng.uniform(1.0, 2.5),
            k=int(rng.choice([1, 2])),
            p1=rng.uniform(0, 2 * np.pi), p2=rng.uniform(0, 2 * np.pi),
            p3=rng.uniform(0, 2 * np.pi),
            rad=rng.uniform(5.2, 8.5),
            peak=rng.uniform(0.16, 0.28),
            mix=rng.uniform(0.0, 1.0),
        ))
    return motes


MOTES = build_motes()


def apply_dust(img, alpha, ph, face, panes):
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
        g = np.exp(-0.5 * ((yy - y) ** 2 + (xx - x) ** 2) / (m['rad'] * m['rad']))
        col = WARM_A * (1.0 - m['mix']) + WARM_B * m['mix']
        dust[yi - rad:yi + rad + 1, xi - rad:xi + rad + 1] += (g * m['peak'])[..., None] * col
    op = 0.70
    base = img.astype(np.float32)
    fur = cv2.dilate((alpha > 0.35).astype(np.uint8),
                     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))).astype(np.float32)
    fur = cv2.GaussianBlur(fur, (0, 0), 2.0) * sstep((620.0 - Y) / 24.0)
    prot = np.clip(np.maximum.reduce([face, fur, panes]), 0, 1)
    add = np.clip(dust * op * (1.0 - prot[..., None]), 0, 255)
    out = 255.0 - (255.0 - base) * (1.0 - add / 255.0)
    return np.clip(out, 0, 255)


def process_one(dst, donor, cat_a, shots, face, panes, ph, title_src):
    img = restore_snow(dst, donor, cat_a)
    img = decontam_crown(img, cat_a)
    img = apply_screen(img, shots, ph)
    img = apply_dust(img, cat_a, ph, face, panes)
    y0, y1, x0, x1 = TITLE
    img = np.clip(img, 0, 255).astype(np.uint8)
    img = img.copy()
    img[y0:y1, x0:x1] = title_src[y0:y1, x0:x1]
    return img


def process(src_o, src_l, mattes, shots, face, panes, preview=None):
    want = set(preview) if preview else set(range(NFR))
    out = [None] * NFR
    for i in range(NFR):
        if i not in want:
            continue
        cat = mattes[i].astype(np.float32)
        if cat.max() > 1.5:
            cat = cat / 255.0
        ph = phase_of(i)
        out[i] = process_one(src_o[i], src_l[i], cat, shots, face, panes, ph, src_o[i])
        if preview or i % 25 == 0:
            Yi = luma(out[i])
            print(f'  n{i} ph{ph} glob={Yi.mean():.1f} scr={Yi[400:530,20:160].mean():.1f}/'
                  f'{Yi[400:530,20:160].max():.0f} bulb={Yi[155:185,300:330].max():.0f}',
                  flush=True)
    return out


def save_preview(out, idx, dest):
    dest.mkdir(parents=True, exist_ok=True)
    crops = {
        'head': (120, 250, 450, 860),
        'crown': (125, 200, 450, 820),
        'lamp': (90, 210, 220, 460),
        'screen': (360, 580, 0, 230),
        'dust': (250, 520, 300, 700),
        'sill': (280, 420, 250, 860),
        'window': (0, 420, 250, 860),
        'tail': (480, 720, 880, 1280),
    }
    thumbs, labs = [], []
    for i in idx:
        if out[i] is None:
            continue
        cv2.imwrite(str(dest / f'full-n{i}.png'), out[i])
        for tag, (y0, y1, x0, x1) in crops.items():
            c = out[i][y0:y1, x0:x1]
            z = cv2.resize(c, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(str(dest / f'{tag}-3x-n{i}.jpg'), z, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        t = cv2.resize(out[i], (640, 360))
        cv2.putText(t, f'n{i}', (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (240, 240, 240), 2)
        thumbs.append(t)
        labs.append(i)
    if thumbs:
        while len(thumbs) % 4:
            thumbs.append(np.zeros((360, 640, 3), np.uint8))
        grid = np.vstack([np.hstack(thumbs[r * 4:(r + 1) * 4]) for r in range(len(thumbs) // 4)])
        cv2.imwrite(str(dest / 'contact-sheet.jpg'), grid, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
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
        ('v41o-ex-mp4', P / 'v41o-ex-test-25s.mp4', V41O_SHA),
        ('v41o-ex-los', SRC_O, V41O_LOS_SHA),
        ('v41l-los', SRC_L, V41L_LOS_SHA),
        ('pcm', PCM, Hh.PCM_SHA),
    ):
        got = sha256(path)
        if got != expect:
            raise SystemExit(f'{label} sha changed: {got}')
    print('frozen shas ok', flush=True)

    shots = load_cartoons()
    print('loading v41o-ex and v41l lossless…', flush=True)
    src_o = Hh.load_span(SRC_O)
    src_l = Hh.load_span(SRC_L)
    assert len(src_o) == NFR and len(src_l) == NFR
    print('loading cat mattes…', flush=True)
    mattes = Hh.load_span(MATTE, gray=True)
    face = G.face_protect()
    panes = G.pane_solid()

    if args.preview:
        idx = [0, 6, 30, 60, 88, 89, 138, 241, 370, 432]
        out = process(src_o, src_l, mattes, shots, face, panes, preview=idx)
        save_preview(out, idx, QC / 'preview')
        return

    out = process(src_o, src_l, mattes, shots, face, panes)
    y0, y1, x0, x1 = TITLE
    bad = [i for i, (a, b) in enumerate(zip(src_o, out))
           if not np.array_equal(a[y0:y1, x0:x1], b[y0:y1, x0:x1])]
    print('title mismatches' if bad else 'title core byte-identical', len(bad) if bad else '', flush=True)

    WORK.mkdir(exist_ok=True)
    los = WORK / f'{VER}-span-890-915-lossless.mkv'
    print('writing lossless…', flush=True)
    Hh.write_lossless(out, los)
    print('encoding mp4…', flush=True)
    Hh.encode_mp4(out, mp4)
    print(VER, 'mp4', sha256(mp4), flush=True)
    print(VER, 'lossless', sha256(los), flush=True)
    shutil.copy2(mp4, WORK / f'{VER}-test-25s.mp4')
    shutil.copy2(los, WORK / f'{VER}-span-890-915-lossless.mkv')
    shutil.copy2(Path(__file__), WORK / 'build_v41p_ex.py')
    print('v41 still', sha256(Hh.V41_MP4), flush=True)
    print('v41o-ex still', sha256(P / 'v41o-ex-test-25s.mp4'), flush=True)


if __name__ == '__main__':
    main()
