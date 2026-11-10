#!/usr/bin/env python3
"""v41l: post-pass on frozen v41k.

Fixes the pale cyan crown/ear halo, the bluish lamp-rim crescent, the
overexposed sticker-like laptop screen, and too-dim dust.

Does not overwrite v41k or earlier named encodes. Audio is the approved
v41 stem (mux only). Title is hard-copied from the v41k source frames.
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

VER = 'v41l'
QC = P / 'qc' / VER
for d in ('preview', 'encoded'):
    (QC / d).mkdir(parents=True, exist_ok=True)
WORK = P / VER
WORK.mkdir(exist_ok=True)
(WORK / 'executor-encode').mkdir(exist_ok=True)

SRC = P / 'v41k' / 'executor-encode' / 'v41k-span-890-915-lossless.mkv'
PCM = P / 'work' / 'test-audio-890-915-v41.f32le'
MATTE = P / 'v41f' / 'cat-alpha-890-915.mkv'
CARTOON_DIR = P / 'v41k' / 'cartoons'
SHOT_PATHS = [
    CARTOON_DIR / 'shot-a-yarn-pounce.jpg',
    CARTOON_DIR / 'shot-b-yarn-chase.jpg',
    CARTOON_DIR / 'shot-c-bird-watch.jpg',
    CARTOON_DIR / 'shot-d-yarn-bat.jpg',
]

NFR, W, H, FPS, NCYCLE = 600, 1280, 720, 24, 241
TITLE = Hh.TITLE
Y, X = R.Y, R.X
sstep, luma, phase_of = Hh.sstep, Hh.luma, Hh.phase_of

V41_MP4, V41_SHA = P / 'v41-test-25s.mp4', 'f53cb1127c75775b740b555dd856e0002c784f80a2a3097118d8e197a794e95f'
V41K_MP4 = P / 'v41k' / 'executor-encode' / 'v41k-test-25s.mp4'
V41K_SHA = '37679567f1e4cb08c0029bc36e9f139e4b9ea6b9744c745d0a4312680219bdf4'
V41K_LOS_SHA = '092fe8a765bb2e7e7a38ce308c27468a5ec88ffb32f917662bc8c6914a926a60'
PCM_SHA = 'a635074f164c14fb3faabbf33c258703e9b7b9645eddbf8028f63a29573745a0'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for ch in iter(lambda: f.read(1 << 20), b''):
            h.update(ch)
    return h.hexdigest()


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


# ---------- 1. crown / ear cyan halo ----------
def decontam_crown(img, cat_a):
    """Choke the matte 1–2 px and pull fur colour outward. No blur.

    Only the silhouette against the window (crown and ears).
    """
    # window-adjacent head band
    win = sstep((260.0 - Y) / 16.0) * sstep((X - 430.0) / 14.0) * sstep((900.0 - X) / 16.0)
    if float(win.max()) < 0.1:
        return img
    a = np.clip(cat_a, 0, 1).astype(np.float32)
    ker1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    ker2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # ~2 px
    ker8 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    solid = (a > 0.50).astype(np.uint8)
    # 1 px choke so thin ear tips survive; protect leftover 1-px peaks
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
    pale = ((B - Rc) > 2.5) & (Yv > 24.0) & (Yv < 90.0) & (warm < 6.0) & (near > 0)
    # Cool contamination sitting on the crown / ear tufts themselves
    edge = cv2.dilate(solid, ker2) & (cv2.erode(solid, ker2) == 0)
    cool_on_fur = (edge > 0) & (win > 0.15) & (warm < 12.0) & (Yv < 95.0) & (Yv > 20.0)
    fur_edge = cv2.dilate(sure, ker2)
    sky_zone = ((near > 0) | pale) & (fur_edge == 0) & (win > 0.15)
    over = ((fur_edge > 0) & (sure == 0) & (win > 0.15)) | (cool_on_fur & (fur_edge > 0))
    if not (sky_zone.any() or over.any()):
        return img
    out = img.copy()
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


# ---------- 2. lamp rim: recolour bluish-white to brass / shadow ----------
HOOD_POLY = np.array([
    [215, 145], [215, 130], [225, 85], [250, 52], [280, 40], [320, 36],
    [355, 44], [380, 58], [405, 85], [422, 115], [432, 145], [422, 172],
    [400, 190], [360, 202], [320, 206], [280, 202], [245, 190], [225, 165],
], np.int32)


def hood_mask():
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [HOOD_POLY], 255)
    return m


def fix_lamp_rim(img):
    # Work on a tight crop so we don't touch other warm light in the room.
    y0, y1, x0, x1 = 80, 220, 220, 460
    crop = img[y0:y1, x0:x1].copy()
    B, Gch, Rc = crop[..., 0], crop[..., 1], crop[..., 2]
    Yv = 0.114 * B + 0.587 * Gch + 0.299 * Rc
    yy, xx = np.indices(Yv.shape)
    Yg, Xg = yy + y0, xx + x0
    warm = ((Yv > 155) & ((Rc - B) > 30)) | ((Yv > 140) & ((Rc - B) > 100))
    m = cv2.GaussianBlur(warm.astype(np.float32), (0, 0), 3.0)
    bowl = (m > 0.3).astype(np.uint8)
    ker15 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    ker7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    bowl = cv2.morphologyEx(bowl, cv2.MORPH_CLOSE, ker15)
    cnts, _ = cv2.findContours(bowl, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        if len(c) >= 5:
            try:
                ell = np.zeros(bowl.shape, np.uint8)
                cv2.ellipse(ell, cv2.fitEllipse(c), 255, -1)
                bowl = ((bowl > 0) | (ell > 0)).astype(np.uint8)
            except cv2.error:
                pass
    outer = cv2.dilate(bowl, ker7)
    ring = (outer > 0) & (bowl == 0)
    pale = (Yv > 80) & (Yv < 195) & ((Rc - B) < 55) & ((Rc - B) > -8)
    bad = ring & pale & (Xg > 300) & (Xg < 420) & (Yg > 145) & (Yg < 200)
    if not bad.any():
        return img
    # Mid-shadow copper sampled from the clean lower-left lip of the opening.
    col = np.array([40.0, 110.0, 158.0], np.float32)
    alpha = np.clip(cv2.GaussianBlur(bad.astype(np.float32), (0, 0), 0.75), 0, 1)[..., None]
    crop = np.clip(crop * (1.0 - alpha) + col * alpha, 0, 255)
    out = img.copy()
    out[y0:y1, x0:x1] = crop
    return out


# ---------- 3. laptop screen: dim, keystone, bezel, texture ----------
# Inset and slightly receding vs the v41k quad, so the picture sits in the lid.
# Slight inset + a little receding on the right, still inside the real bezel.
SCREEN_SRC = np.array([[17, 387], [143, 389], [189, 547], [17, 547]], np.float32)
SW, SH = 400, 512
SCREEN_DST = np.array([[0, 0], [SW - 1, 0], [SW - 1, SH - 1], [0, SH - 1]], np.float32)
H_INV = cv2.getPerspectiveTransform(SCREEN_DST, SCREEN_SRC)
# just wide enough to hide the old bright sticker
COVER_SRC = np.array([[13, 383], [147, 383], [194, 551], [13, 551]], np.float32)
H_COVER = cv2.getPerspectiveTransform(SCREEN_DST, COVER_SRC)


def screen_feather(src_pts, sigma=1.8, inset=0.0):
    m = np.zeros((H, W), np.float32)
    pts = src_pts.copy()
    cv2.fillPoly(m, [pts.astype(np.int32)], 1.0)
    if inset > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(inset) * 2 + 1, int(inset) * 2 + 1))
        m = cv2.erode(m, k)
    m = cv2.GaussianBlur(m, (0, 0), sigma)
    return np.clip(m, 0, 1)


def grade_rect(rect):
    """Dim to the dusk room, add vignette, scanlines, a touch of glass bloom."""
    r = rect.astype(np.float32)
    # dim so peak sits well below the lamp bulb (~240). Target peak ~145, mean ~95.
    r = r * 0.55
    # slight warm grade (dusk room, not a daylight panel)
    r = r * np.array([0.90, 0.96, 1.02], np.float32)
    # vignette
    yy, xx = np.mgrid[:SH, :SW].astype(np.float32)
    cy, cx = (SH - 1) / 2.0, (SW - 1) / 2.0
    rho = np.sqrt(((yy - cy) / (SH * 0.62)) ** 2 + ((xx - cx) / (SW * 0.62)) ** 2)
    vig = 1.0 - 0.28 * np.clip(rho, 0, 1) ** 2
    r = r * vig[..., None]
    # faint scanlines (gentle LCD texture, not a heavy CRT)
    scan = 1.0 - 0.028 * (0.5 + 0.5 * np.sin(yy * np.pi / 2.2))
    r = r * scan[..., None]
    r = cv2.GaussianBlur(r, (0, 0), 0.50)
    # soft bloom of the content (screen glow)
    bloom = cv2.GaussianBlur(r, (0, 0), 2.8)
    r = np.clip(r * 0.88 + bloom * 0.18, 0, 255)
    return r


def apply_screen(frame, shots, ph):
    si, local, length = shot_of(ph)
    rect = grade_rect(render_shot(shots[si], local, length, SW, SH))
    pictured = cv2.warpPerspective(
        np.clip(rect, 0, 255).astype(np.uint8), H_INV, (W, H),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
    ).astype(np.float32)
    # cover mask hides the old bright sticker; picture mask is the inset screen
    cover = screen_feather(COVER_SRC, sigma=1.6)
    picture = screen_feather(SCREEN_SRC, sigma=1.4)
    img = frame.astype(np.float32)
    # sample a dark bezel colour from the existing lid, just outside the opening
    bezel_band = np.clip(cover - picture, 0, 1)
    # fallback dark bezel
    bezel_col = np.array([22.0, 24.0, 26.0], np.float32)
    wts = bezel_band * sstep((40.0 - luma(img)) / 12.0)
    if float(wts.sum()) > 20:
        bezel_col = (img * wts[..., None]).sum(axis=(0, 1)) / (wts.sum() + 1e-6)
        bezel_col = np.clip(bezel_col, 10, 50)
    # 1) paint dark bezel over the old bright edges
    img = img * (1.0 - cover[..., None]) + bezel_col * cover[..., None]
    # 2) drop in the graded picture
    img = img * (1.0 - picture[..., None]) + pictured * picture[..., None]
    # 3) thin contact shadow just inside the bezel (darken the screen lip)
    edge = cv2.dilate(picture, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    edge = np.clip(edge - cv2.erode(picture, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))), 0, 1)
    edge = cv2.GaussianBlur(edge, (0, 0), 1.1)
    img = img * (1.0 - 0.38 * edge[..., None])
    # 4) little warm glow on the bezel around the opening
    glow = cv2.GaussianBlur(picture, (0, 0), 4.5)
    glow = np.clip(glow - picture, 0, 1)
    mean_c = pictured.reshape(-1, 3).mean(axis=0) * 0.35
    img = img + glow[..., None] * mean_c * 0.55
    return np.clip(img, 0, 255)


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
    margin = 18
    tw, th = sw + 2 * margin, sh + 2 * margin
    canvas = cv2.resize(im, (tw, th), interpolation=cv2.INTER_AREA).astype(np.float32)
    u = local / max(length - 1, 1)
    dx = 10.0 * np.sin(2 * np.pi * u) + 4.0 * u
    dy = 8.0 * np.sin(2 * np.pi * u + 0.7) - 3.0 * np.cos(np.pi * u)
    x0 = int(np.clip(round(margin + dx), 0, tw - sw))
    y0 = int(np.clip(round(margin + dy), 0, th - sh))
    return canvas[y0:y0 + sh, x0:x0 + sw]


# ---------- 4. warmer, more visible soft dust ----------
WARM_A = np.array([0xC0, 0xE6, 0xFF], np.float32)  # #FFE6C0
WARM_B = np.array([0x90, 0xC8, 0xE8], np.float32)  # #E8C890


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


def process_one(frame, cat_a, shots, face, panes, ph, title_src):
    img = frame.astype(np.float32)
    img = decontam_crown(img, cat_a)
    img = fix_lamp_rim(img)
    img = apply_screen(img, shots, ph)
    img = apply_dust(img, cat_a, ph, face, panes)
    img = np.clip(img, 0, 255).astype(np.uint8)
    y0, y1, x0, x1 = TITLE
    img[y0:y1, x0:x1] = title_src[y0:y1, x0:x1]
    return img


def process(src, mattes, shots, face, panes, preview=None):
    want = set(preview) if preview else set(range(NFR))
    out = [None] * NFR
    for i in range(NFR):
        if i not in want:
            continue
        cat = mattes[i].astype(np.float32)
        if cat.max() > 1.5:
            cat = cat / 255.0
        ph = phase_of(i)
        out[i] = process_one(src[i], cat, shots, face, panes, ph, src[i])
        if preview or i % 25 == 0:
            Yi = luma(out[i])
            print(f'  n{i} ph{ph} glob={Yi.mean():.1f} scr={Yi[400:530,20:160].mean():.1f} '
                  f'bulb={Yi[155:185,300:330].max():.0f}', flush=True)
    return out


def save_preview(out, idx, dest):
    dest.mkdir(parents=True, exist_ok=True)
    crops = {
        'head': (120, 250, 450, 860),
        'crown': (125, 200, 450, 820),
        'lamp': (90, 210, 220, 460),
        'screen': (360, 580, 0, 230),
        'dust': (250, 520, 300, 700),
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
        sheet.append(out[i])
        labs.append(f'n{i}')
    if sheet:
        thumbs = []
        for f, lab in zip(sheet, labs):
            t = cv2.resize(f, (640, 360), interpolation=cv2.INTER_AREA)
            cv2.putText(t, lab, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2, cv2.LINE_AA)
            thumbs.append(t)
        while len(thumbs) % 4:
            thumbs.append(np.zeros((360, 640, 3), np.uint8))
        grid = np.vstack([np.hstack(thumbs[r * 4:(r + 1) * 4]) for r in range(len(thumbs) // 4)])
        cv2.imwrite(str(dest / 'contact-sheet.jpg'), grid, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    print('preview written', dest)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--preview', action='store_true')
    args = ap.parse_args()

    for label, path, expect in (
        ('v41', V41_MP4, V41_SHA),
        ('v41k-mp4', V41K_MP4, V41K_SHA),
        ('v41k-los', SRC, V41K_LOS_SHA),
        ('pcm', PCM, PCM_SHA),
    ):
        got = sha256(path)
        if got != expect:
            raise SystemExit(f'{label} sha changed: {got}')
    print('frozen shas ok', flush=True)

    shots = load_cartoons()
    print('loading v41k lossless…', flush=True)
    src = R.load_span(SRC)
    assert len(src) == NFR, len(src)
    print('loading cat mattes…', flush=True)
    mattes = R.load_span(MATTE, gray=True)
    assert len(mattes) == NFR, len(mattes)
    face = G.face_protect()
    panes = G.pane_solid()

    if args.preview:
        idx = [0, 30, 60, 88, 89, 90, 138, 199, 241, 370, 432]
        out = process(src, mattes, shots, face, panes, preview=idx)
        save_preview(out, idx, QC / 'preview')
        return

    out = process(src, mattes, shots, face, panes)
    y0, y1, x0, x1 = TITLE
    bad = [i for i, (a, b) in enumerate(zip(src, out))
           if not np.array_equal(a[y0:y1, x0:x1], b[y0:y1, x0:x1])]
    print('title mismatches' if bad else 'title core byte-identical', len(bad) if bad else '', flush=True)
    los = WORK / 'v41l-span-890-915-lossless.mkv'
    mp4 = P / 'v41l-test-25s.mp4'
    print('writing lossless…', flush=True)
    Hh.write_lossless(out, los)
    print('encoding mp4…', flush=True)
    Hh.encode_mp4(out, mp4)
    print('v41l mp4', sha256(mp4), flush=True)
    print('v41l lossless', sha256(los), flush=True)
    enc = WORK / 'executor-encode'
    shutil.copy2(mp4, enc / 'v41l-test-25s.mp4')
    shutil.copy2(los, enc / 'v41l-span-890-915-lossless.mkv')
    shutil.copy2(Path(__file__), enc / 'build_v41l.py')
    for label, path in (('v41', V41_MP4), ('v41k-mp4', V41K_MP4), ('v41k-los', SRC)):
        print(label, 'still', sha256(path), flush=True)


if __name__ == '__main__':
    main()
