#!/usr/bin/env python3
"""v41q-ex: polish post-pass on frozen v41p-ex.

Keeps v41p-ex dusk sky, restored snow content, room darkening, tail,
crown cleanup, full-bezel cartoon and dust. Grades the pasted frost bank
and snowy hill into the dusk (cooler, darker, a slight lavender afterglow,
soft feathered top), removes the greenish/grey smear next to the indoor
plant, cleans the residual left-ear-tip fringe through the head-turn, and
dims the laptop another ~15% with a warm room reflection and soft
gradient so it reads as a lit screen.

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
import build_v41h as Hh

VER = 'v41q-ex'
WORK = P / VER
QC = P / 'qc' / VER
SRC = P / 'v41p-ex' / 'v41p-ex-span-890-915-lossless.mkv'
MATTE = Hh.MATTE
PCM = Hh.PCM

V41P_SHA = 'c1ebb58d95804315b65c7a1f0547df27e073086353dc948c05e765a119a0dea7'
V41P_LOS_SHA = 'e2bae6bfe1fb92efe0de085d12c4a3fdfc296cfd532285e04ecd9784b0273141'

NFR, W, H, NCYCLE = 600, 1280, 720, 241
TITLE = Hh.TITLE
Yg, Xg = np.mgrid[0:H, 0:W].astype(np.float32)

PANE_L = np.array([[266, 0], [558, 0], [556, 400], [268, 400]], np.int32)
SCR_Q = np.array([[-2, 377], [161, 377], [193, 554], [-2, 590]], np.float32)

_PANE = None
_SCRM = None


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


def nearest_fill(color, sure, iters=14):
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


def fill_roi(imgf, sure, y0, y1, x0, x1, iters=14):
    filled = nearest_fill(imgf[y0:y1, x0:x1], sure[y0:y1, x0:x1], iters=iters)
    out = imgf.copy()
    out[y0:y1, x0:x1] = filled
    return out


def pane_l():
    global _PANE
    if _PANE is None:
        m = np.zeros((H, W), np.uint8)
        cv2.fillPoly(m, [PANE_L], 1)
        _PANE = m
    return _PANE


def scr_mask():
    global _SCRM
    if _SCRM is None:
        big = np.zeros((H * 4, W * 4), np.uint8)
        cv2.fillPoly(big, [np.round(SCR_Q * 4).astype(np.int32)], 255)
        m = cv2.resize(big, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        _SCRM = cv2.GaussianBlur(m, (0, 0), 0.8)
    return _SCRM


def grade_frost_and_smear(img):
    """Fit the frost/hill into the dusk grade and lift the plant-side smear."""
    imgf = img.astype(np.float32)
    Yv = luma(imgf)
    B, G, R = imgf[..., 0], imgf[..., 1], imgf[..., 2]
    chroma = np.maximum(np.maximum(R, G), B) - np.minimum(np.minimum(R, G), B)
    open_g = pane_l() > 0
    plant = ((R > 42) & (G > 42) & (B < 58) & (((R + G) * 0.5 - B) > 30)
             & (Yg > 230) & (Yg < 400) & (Xg < 380))
    plant_core = plant.astype(np.uint8)
    plant_d = cv2.dilate(plant_core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    plant_protect = cv2.dilate(plant_core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    # Tight smear box next to the indoor plant. Require a greenish/grey
    # cast so the adjacent frost lip is not inpainted away.
    smear_zone = (open_g & (Xg >= 275) & (Xg <= 345) & (Yg >= 270) & (Yg <= 312)
                  & (plant_core == 0))
    greenish = ((G - B) > 14) & ((G - B) < 70)
    muddy = smear_zone & (Yv >= 28) & (Yv <= 100) & (chroma < 55) & greenish
    mint = smear_zone & (Yv >= 24) & (Yv <= 110) & ((G - B) > 16) & ((G - B) < 60) & (chroma < 65)
    near_plant = cv2.dilate(plant_core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    halo = (smear_zone & (near_plant > 0) & (Yv >= 24) & (Yv <= 85)
            & greenish & (chroma < 50) & (plant_core == 0))
    smear = ((muddy | mint | halo) & (plant_protect == 0)).astype(np.uint8)
    smear = cv2.morphologyEx(smear, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    smear = smear & (plant_protect == 0) & open_g.astype(np.uint8)
    if smear.any():
        sure = (open_g & (plant_protect == 0) & (smear == 0) & (Yv < 26)
                & (Xg < 400) & (Yg >= 200) & (Yg <= 340))
        skyish = fill_roi(imgf, sure, 200, 360, 250, 420, iters=16)
        m = np.clip(cv2.GaussianBlur(smear.astype(np.float32), (0, 0), 1.3), 0, 1)
        m[plant_protect > 0] = 0
        imgf = imgf * (1.0 - m[..., None]) + skyish * m[..., None]
        Yv = luma(imgf)
        B, G, R = imgf[..., 0], imgf[..., 1], imgf[..., 2]
        chroma = np.maximum(np.maximum(R, G), B) - np.minimum(np.minimum(R, G), B)

    roi = open_g & (Xg >= 262) & (Xg <= 535) & (Yg >= 238) & (Yg <= 340)
    raw = (roi & (plant_d == 0) & (Yv > 24) & (chroma < 80)).astype(np.uint8)
    frost = cv2.morphologyEx(raw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    frost = cv2.morphologyEx(frost, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    # Do not invent a rectangular slab; stay close to the real bank.
    frost = frost & cv2.dilate(raw, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    frost = frost & roi.astype(np.uint8) & (plant_d == 0)
    nlab, lab, stats, _ = cv2.connectedComponentsWithStats(frost, 8)
    keep = np.zeros_like(frost)
    for i in range(1, nlab):
        area = int(stats[i, cv2.CC_STAT_AREA])
        x0 = int(stats[i, cv2.CC_STAT_LEFT])
        if area >= 180 and x0 >= 300:
            keep[lab == i] = 1
        elif area >= 400:
            keep[lab == i] = 1
    frost = keep
    if not frost.any():
        return np.clip(imgf, 0, 255)

    rows = np.arange(H)[:, None]
    top = np.where(frost > 0, rows, H).min(axis=0)
    dist = rows - top[None, :]
    dist = np.where((top[None, :] < H) & (dist >= 0) & (dist < 80), dist, 80).astype(np.float32)

    tree_sure = (open_g & (plant_d == 0) & (frost == 0) & (Yv < 24)
                 & (Yg >= 180) & (Yg <= 300) & (Xg >= 260) & (Xg <= 540))
    tree_col = fill_roi(imgf, tree_sure, 170, 360, 250, 550, iters=14)

    # Darker, lower contrast, cooler, with a lavender afterglow cast.
    Yn = (Yv - 40.0) * 0.50 + 26.0
    scale = np.clip(np.where(Yv > 1.0, Yn / np.maximum(Yv, 1.0), 1.0), 0.40, 1.0)
    graded = imgf * scale[..., None]
    graded = graded * np.array([1.14, 0.88, 1.16], np.float32)
    graded = graded + np.array([6.0, 3.0, 9.0], np.float32)
    graded = np.clip(graded, 0, 255)

    m = frost.astype(np.float32)
    m = np.clip(cv2.GaussianBlur(m, (0, 0), 2.6), 0, 1)
    m[plant_core > 0] = 0
    crest = sstep((22.0 - dist) / 22.0) * (frost > 0).astype(np.float32)
    crest = np.clip(cv2.GaussianBlur(crest, (0, 0), 2.2), 0, 1)
    snow_mix = graded * (1.0 - 0.65 * crest[..., None]) + tree_col * (0.65 * crest[..., None])
    snow_mix = snow_mix * (1.0 - 0.10 * crest)[..., None]
    out = imgf * (1.0 - m[..., None]) + snow_mix * m[..., None]
    return np.clip(out, 0, 255)


def clean_ear(img, cat_a):
    """Clear residual dark/cool fringe on the crown and left ear tip."""
    a = cat_a.astype(np.float32)
    if a.max() > 1.5:
        a = a / 255.0
    a = np.clip(a, 0, 1)
    win = sstep((255.0 - Yg) / 20.0) * sstep((Xg - 420.0) / 14.0) * sstep((920.0 - Xg) / 16.0)
    if float(win.max()) < 0.1:
        return img
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
    y0, y1, x0, x1 = 110, 270, 420, 920
    fur = fill_roi(img.astype(np.float32), sure.astype(bool), y0, y1, x0, x1, iters=16)
    sky_seed = (near == 0) & (win > 0.35) & (a < 0.08)
    if int(sky_seed.sum()) < 20:
        sky_seed = (win > 0.5) & (a < 0.05)
    sky = fill_roi(img.astype(np.float32), sky_seed.astype(bool), y0, y1, x0, x1, iters=16)
    Yv = luma(img)
    B, Gch, Rc = img[..., 0].astype(np.float32), img[..., 1].astype(np.float32), img[..., 2].astype(np.float32)
    warm = Rc - B
    chroma = np.maximum(np.maximum(Rc, Gch), B) - np.minimum(np.minimum(Rc, Gch), B)
    pale = ((B - Rc) > 2.0) & (Yv > 22) & (Yv < 95) & (warm < 8) & (near > 0)
    dark = (Yv < 46) & (warm < 28) & (chroma < 32) & (near > 0) & (sure == 0)
    fringe = (a > 0.02) & (a < 0.45) & (Yv < 48) & (warm < 30) & (near > 0)
    edge = cv2.dilate(solid, ker2) & (cv2.erode(solid, ker2) == 0)
    cool_on_fur = (edge > 0) & (win > 0.10) & (warm < 16) & (Yv < 110)
    fur_ring = cv2.dilate(sure, ker2) & (sure == 0)
    sky_zone = (pale | dark | fringe) & (fur_ring == 0) & (win > 0.10) & (warm < 28)
    over = ((fur_ring > 0) & (win > 0.10)) | (cool_on_fur & (fur_ring > 0))
    out = img.copy().astype(np.float32)
    if sky_zone.any():
        m = np.clip(cv2.GaussianBlur(sky_zone.astype(np.float32), (0, 0), 0.45), 0, 1)
        m[sure > 0] = 0
        out = out * (1.0 - m[..., None]) + sky * m[..., None]
    if over.any():
        m = np.clip(cv2.GaussianBlur(over.astype(np.float32), (0, 0), 0.35), 0, 1)
        out = out * (1.0 - m[..., None]) + fur * m[..., None]
    return np.clip(out, 0, 255)


def grade_screen(img):
    """Another ~15% down, plus a warm glass reflection and a soft LCD falloff."""
    a = scr_mask()
    imgf = img.astype(np.float32)
    graded = imgf * 0.80
    mid = 64.0
    graded = np.clip(mid + 1.08 * (graded - mid), 0, 255)
    tx = np.clip(Xg / 170.0, 0, 1.4)
    ty = np.clip((Yg - 377.0) / 200.0, -0.1, 1.2)
    rho = np.sqrt(((Xg - 90.0) / 125.0) ** 2 + ((Yg - 450.0) / 155.0) ** 2)
    fall = 1.0 - 0.18 * np.clip(rho, 0, 1.25) ** 1.35
    vert = 1.0 + 0.06 * (1.0 - np.clip(ty, 0, 1)) - 0.06 * np.clip(ty, 0, 1)
    graded = graded * (fall * vert)[..., None]
    warm = np.array([42.0, 140.0, 210.0], np.float32)
    wash = 0.045 * sstep((0.95 - ty) / 0.85) * sstep((tx + 0.05) / 0.8)
    blob = np.exp(-0.5 * (((Xg - 115.0) / 65.0) ** 2 + ((Yg - 415.0) / 50.0) ** 2))
    refl = np.clip(wash + 0.055 * blob, 0, 0.12)
    Yv = luma(graded)
    glass = 0.50 + 0.50 * np.clip(1.0 - Yv / 140.0, 0, 1)
    graded = np.clip(graded + (refl * glass)[..., None] * warm, 0, 255)
    out = imgf * (1.0 - a[..., None]) + graded * a[..., None]
    return np.clip(out, 0, 255)


def process_one(src, cat_a):
    img = grade_frost_and_smear(src)
    img = clean_ear(img, cat_a)
    img = grade_screen(img)
    img = np.clip(img, 0, 255).astype(np.uint8)
    y0, y1, x0, x1 = TITLE
    img = img.copy()
    img[y0:y1, x0:x1] = src[y0:y1, x0:x1]
    return img


def process(src, mattes, preview=None):
    want = set(preview) if preview else set(range(NFR))
    out = [None] * NFR
    for i in range(NFR):
        if i not in want:
            continue
        cat = mattes[i].astype(np.float32)
        if cat.max() > 1.5:
            cat = cat / 255.0
        out[i] = process_one(src[i], cat)
        if preview or i % 25 == 0:
            Yi = luma(out[i])
            print(f'  n{i} glob={Yi.mean():.1f} scr={Yi[400:530, 20:160].mean():.1f}/'
                  f'{Yi[400:530, 20:160].max():.0f} bulb={Yi[155:185, 300:330].max():.0f}',
                  flush=True)
    return out


def save_preview(out, idx, dest):
    dest.mkdir(parents=True, exist_ok=True)
    crops = {
        'head': (120, 250, 450, 860),
        'crown': (125, 200, 450, 820),
        'ear': (140, 210, 500, 680),
        'lamp': (90, 210, 220, 460),
        'screen': (360, 580, 0, 230),
        'dust': (250, 520, 300, 700),
        'sill': (280, 420, 250, 860),
        'window': (0, 420, 250, 860),
        'frost': (230, 360, 250, 520),
        'smear': (260, 320, 260, 360),
        'tail': (480, 720, 880, 1280),
    }
    thumbs = []
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
        ('v41p-ex-mp4', P / 'v41p-ex-test-25s.mp4', V41P_SHA),
        ('v41p-ex-los', SRC, V41P_LOS_SHA),
        ('pcm', PCM, Hh.PCM_SHA),
    ):
        got = sha256(path)
        if got != expect:
            raise SystemExit(f'{label} sha changed: {got}')
    print('frozen shas ok', flush=True)

    print('loading v41p-ex lossless…', flush=True)
    src = Hh.load_span(SRC)
    assert len(src) == NFR
    print('loading cat mattes…', flush=True)
    mattes = Hh.load_span(MATTE, gray=True)

    if args.preview:
        idx = [0, 6, 88, 89, 138, 241, 350, 360, 368, 370, 372, 380, 432]
        out = process(src, mattes, preview=idx)
        save_preview(out, idx, QC / 'preview')
        save_preview(out, idx, WORK / 'preview')
        return

    out = process(src, mattes)
    y0, y1, x0, x1 = TITLE
    bad = [i for i, (a, b) in enumerate(zip(src, out))
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
    shutil.copy2(Path(__file__), WORK / 'build_v41q_ex.py')
    print('v41 still', sha256(Hh.V41_MP4), flush=True)
    print('v41p-ex still', sha256(P / 'v41p-ex-test-25s.mp4'), flush=True)


if __name__ == '__main__':
    main()
