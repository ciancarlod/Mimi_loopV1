#!/usr/bin/env python3
"""v41r-ex: post-pass on frozen v41q-ex.

Fixes the meow mouth and the flattened/blurred ears, and removes the
baked-in dust motes entirely (steam stays). Mouth and ear pixels come
from frozen v41, colour-matched to the dusk/lamp head; no fur puppet-warp.
Ears use the rembg cat matte as a soft-but-tight alpha, with fur colour
pulled outward (not a shape choke), and follow the head rigidly because
they are taken from the same frame of v41.

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

VER = 'v41r-ex'
WORK = P / VER
QC = P / 'qc' / VER
SRC_Q = P / 'v41q-ex' / 'v41q-ex-span-890-915-lossless.mkv'
SRC_V = P / 'work' / 'v41-span-890-915-lossless.mkv'
MATTE = Hh.MATTE
PCM = Hh.PCM

V41Q_SHA = 'be65528916d68b99cc64546d865388f824656b3975adb8f861f10d03e548afcd'
V41Q_LOS_SHA = '9e62b006a28977f54ac5f0a0db9e25b118c1837922cd6796e9d032ad0b0dfe6e'
V41_LOS_SHA = '226206bf77a1e30bb4e9daf44d9885897edf3079f1043376c5d109bc4025f796'

NFR, W, H, NCYCLE = 600, 1280, 720, 241
TITLE = Hh.TITLE
Yg, Xg = np.mgrid[0:H, 0:W].astype(np.float32)

SCR_Q = np.array([[-2, 377], [161, 377], [193, 554], [-2, 590]], np.float32)
PANE_L = np.array([[266, 0], [558, 0], [556, 400], [268, 400]], np.int32)
PANE_R = np.array([[600, 0], [840, 0], [838, 400], [600, 400]], np.int32)

_PANE = None
_SCR = None


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
    global _PANE
    if _PANE is None:
        m = np.zeros((H, W), np.uint8)
        cv2.fillPoly(m, [PANE_L, PANE_R], 1)
        _PANE = m
    return _PANE


def scr_mask_bin():
    global _SCR
    if _SCR is None:
        m = np.zeros((H, W), np.uint8)
        cv2.fillPoly(m, [np.round(SCR_Q).astype(np.int32)], 1)
        _SCR = m
    return _SCR


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


def affine_match(src, dst, mask):
    a = np.ones(3, np.float32)
    b = np.zeros(3, np.float32)
    if int(mask.sum()) < 40:
        return a, b
    for c in range(3):
        s = src[..., c][mask]
        d = dst[..., c][mask]
        s1, s2 = np.percentile(s, [20, 80])
        d1, d2 = np.percentile(d, [20, 80])
        if s2 - s1 < 3:
            a[c] = float(np.median(d) / max(float(np.median(s)), 1.0))
            b[c] = 0.0
        else:
            a[c] = float((d2 - d1) / (s2 - s1))
            b[c] = float(d1 - a[c] * s1)
        a[c] = float(np.clip(a[c], 0.35, 1.15))
        b[c] = float(np.clip(b[c], -40, 40))
    return a, b


def apply_ab(im, a, b):
    return np.clip(im.astype(np.float32) * a + b, 0, 255)


def norm_alpha(cat_a):
    a = cat_a.astype(np.float32)
    if a.max() > 1.5:
        a = a / 255.0
    return np.clip(a, 0, 1)


def remove_dust(img, cat_a):
    """Remove compact warm motes. Leave steam, lamp, title, glass, fur."""
    imgf = img.astype(np.float32)
    Yv = luma(imgf)
    B, G, R = imgf[..., 0], imgf[..., 1], imgf[..., 2]
    a = norm_alpha(cat_a)
    cat = cv2.dilate((a > 0.12).astype(np.uint8),
                     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    lamp = ((Xg > 200) & (Xg < 460) & (Yg > 70) & (Yg < 220) & (Yv > 140)).astype(np.uint8)
    lamp = cv2.dilate(lamp, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    title = np.zeros((H, W), np.uint8)
    y0, y1, x0, x1 = TITLE
    title[y0:y1, x0:x1] = 1
    air = (cat == 0) & (pane_mask() == 0) & (scr_mask_bin() == 0) & (lamp == 0) & (title == 0)
    med = cv2.medianBlur(np.clip(imgf, 0, 255).astype(np.uint8), 21).astype(np.float32)
    res = luma(imgf) - luma(med)
    warm = (R - B) > 8
    motes = air & warm & (res > 8) & (Yv > 40) & (Yv < 230)
    motes = motes.astype(np.uint8)
    nlab, lab, stats, _ = cv2.connectedComponentsWithStats(motes, 8)
    keep = np.zeros_like(motes)
    for i in range(1, nlab):
        area = int(stats[i, cv2.CC_STAT_AREA])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        if 6 <= area <= 500 and bw <= 36 and bh <= 36 and max(bw, bh) <= 2.4 * min(bw, bh) + 2:
            keep[lab == i] = 1
    if not keep.any():
        return np.clip(imgf, 0, 255)
    keep = cv2.dilate(keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    m = np.clip(cv2.GaussianBlur(keep.astype(np.float32), (0, 0), 1.6), 0, 1)
    out = imgf * (1.0 - m[..., None]) + med * m[..., None]
    return np.clip(out, 0, 255)


def restore_ears(img, donor, cat_a):
    """v41 ear shape and tufts, graded to the dusk head, soft-but-tight alpha.

    Windows stay tight around each ear so the forehead is not re-graded.
    Alpha is rembg (no choke) plus a short fur-coloured overshoot for tufts
    that rembg missed. Sky, ivy and the window frame are never replaced
    by a loose warmth detector.
    """
    a = norm_alpha(cat_a)
    imgf = img.astype(np.float32)
    don = donor.astype(np.float32)
    out = imgf.copy()
    # (y0,y1,x0,x1) — just the ear, not the blaze / cheek
    wins = [
        (145, 225, 575, 710),
        (175, 280, 800, 940),
    ]
    ker1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    ker3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    ker5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    for (y0, y1, x0, x1) in wins:
        aw = a[y0:y1, x0:x1]
        sub_q = imgf[y0:y1, x0:x1]
        sub_v = don[y0:y1, x0:x1]
        if float(aw.max()) < 0.2:
            continue
        solid = (aw > 0.40).astype(np.uint8)
        if int(solid.sum()) < 40:
            solid = (aw > 0.20).astype(np.uint8)
        sure = cv2.erode(solid, ker1)
        if int(sure.sum()) < 30:
            sure = solid
        # short tuft overshoot: grow only into v41 pixels that match ear fur
        med = np.median(sub_v[sure > 0], axis=0).astype(np.float32)
        dt = cv2.distanceTransform((1 - solid).astype(np.uint8), cv2.DIST_L2, 5)
        cd = np.linalg.norm(sub_v - med, axis=2)
        warm = (sub_v[..., 2] > sub_v[..., 0] + 18) & (sub_v[..., 2] > 60)
        ivy = sub_v[..., 1] > sub_v[..., 2] + 8
        tuft = ((dt > 0) & (dt <= 3.5) & (cd < 65) & warm & (ivy == 0)).astype(np.uint8)
        al = np.clip(np.maximum(aw, 0.95 * tuft.astype(np.float32)), 0, 1)
        # grade on mid-ear fur, not the sky edge
        Yq = luma(sub_q)
        Yv = luma(sub_v)
        grade_m = (sure > 0) & (Yq > 28) & (Yv > 40)
        ga, gb = affine_match(sub_v, sub_q, grade_m)
        graded = apply_ab(sub_v, ga, gb)
        fur = nearest_fill(graded, sure.astype(bool), iters=16)
        al_s = np.clip(cv2.GaussianBlur(al, (0, 0), 0.28), 0, 1)
        yy, xx = np.mgrid[0:y1 - y0, 0:x1 - x0].astype(np.float32)
        feat = (sstep(xx / 10.0) * sstep((x1 - x0 - 1 - xx) / 10.0)
                * sstep(yy / 10.0) * sstep((y1 - y0 - 1 - yy) / 10.0))
        # graded v41 inside the ear; pulled fur only on a short fur-coloured ring
        col = fur.copy()
        inn = al_s > 0.12
        col[inn] = (graded * al_s[..., None] + fur * (1.0 - al_s[..., None]))[inn]
        w = np.clip(al_s * feat, 0, 1)
        w = np.maximum(w, 0.90 * tuft.astype(np.float32) * feat)
        v_is_sky = (np.abs(sub_v[..., 2].astype(np.float32) - sub_v[..., 0]) < 25) & (Yv > 120)
        w = w * (1.0 - 0.95 * v_is_sky.astype(np.float32))
        w = w * (1.0 - 0.95 * ivy.astype(np.float32))
        fringe = (al_s > 0.18) & (Yq < 40)
        w = np.maximum(w, fringe.astype(np.float32) * al_s * feat)
        q_ivy = sub_q[..., 1] > sub_q[..., 2] + 8
        w = w * (1.0 - 0.95 * q_ivy.astype(np.float32))
        sub = sub_q * (1.0 - w[..., None]) + col * w[..., None]
        out[y0:y1, x0:x1] = sub
    return np.clip(out, 0, 255)


def restore_mouth(img, donor):
    """v41 meow mouth, colour-matched and warm, covering the q-ex smear."""
    imgf = img.astype(np.float32)
    don = donor.astype(np.float32)
    y0, y1, x0, x1 = 345, 475, 610, 830
    sub_q = imgf[y0:y1, x0:x1]
    sub_v = don[y0:y1, x0:x1]
    Yv = luma(sub_v)
    B, G, R = sub_v[..., 0], sub_v[..., 1], sub_v[..., 2]
    opening = ((Yv < 110) & (R > G + 8) & (R > B + 5) & (Yv > 22)).astype(np.uint8)
    # the broken mauve/purple blob in q-ex, if any
    Yq = luma(sub_q)
    Bq, Gq, Rq = sub_q[..., 0], sub_q[..., 1], sub_q[..., 2]
    mauve = ((Rq > Gq + 5) & (Bq > Gq * 0.65) & (Yq > 24) & (Yq < 135)
             & (Rq < 190) & (Rq > 50)).astype(np.uint8)
    # keep mauve detection inside a mouth band so a cheek freckle can't match
    yy, xx = np.mgrid[0:y1 - y0, 0:x1 - x0].astype(np.float32)
    mauve[yy < 20] = 0
    mauve[yy > 100] = 0
    mauve[xx < 15] = 0
    mauve[xx > 200] = 0
    ker11 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    ker19 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
    around = cv2.dilate(np.maximum(opening, mauve), ker19).astype(np.float32)
    # a tight chin pad under the opening only (not the outer cheeks)
    if opening.any():
        ys, xs = np.where(opening > 0)
        cx, cy = float(np.median(xs)), float(np.median(ys))
        chin_cy = float(ys.max()) + 12.0
        width = max(22.0, 0.48 * (xs.max() - xs.min() + 6))
    elif mauve.any():
        ys, xs = np.where(mauve > 0)
        cx, cy = float(np.median(xs)), float(np.median(ys))
        chin_cy = float(ys.max()) + 10.0
        width = max(22.0, 0.48 * (xs.max() - xs.min() + 6))
    else:
        # closed mouth: only a small slot under the nose, if we can see it
        nose = ((R > 80) & (R > G + 10) & (R > B + 20) & (Yv > 50) & (Yv < 170) & (yy < 50))
        if nose.any():
            ys, xs = np.where(nose)
            cx, cy = float(np.median(xs)), float(ys.max()) + 16.0
        else:
            cx, cy = 100.0, 38.0
        chin_cy = cy + 18.0
        width = 26.0
        around = np.zeros_like(yy, np.float32)
    mouth_slot = np.exp(-0.5 * (((xx - cx) / (width * 0.80)) ** 2 + ((yy - cy) / 14.0) ** 2))
    chin = np.exp(-0.5 * (((xx - cx) / (width * 0.90)) ** 2 + ((yy - chin_cy) / 16.0) ** 2))
    m = np.clip(np.maximum.reduce([around, mouth_slot, 0.80 * chin]), 0, 1)
    # do not let the mask walk into the far right cheek
    m = m * sstep((165.0 - (xx - cx)) / 10.0)
    m = cv2.GaussianBlur(m, (0, 0), 1.6)
    feat = (sstep(xx / 12.0) * sstep((x1 - x0 - 1 - xx) / 12.0)
            * sstep(yy / 10.0) * sstep((y1 - y0 - 1 - yy) / 12.0))
    m = np.clip(m * feat, 0, 1)
    fur_s = (opening == 0) & (mauve == 0) & (m > 0.05) & (Yq > 30) & (Yv > 50)
    ga, gb = affine_match(sub_v, sub_q, fur_s)
    graded = apply_ab(sub_v, ga, gb)
    # keep the opening warm pink / deep red-brown, never mauve
    if opening.any() or mauve.any():
        op = np.maximum(opening, cv2.erode(mauve, ker11)).astype(np.float32)
        op = cv2.GaussianBlur(op, (0, 0), 0.7)
        g = graded.copy()
        excess_b = np.clip(g[..., 0] - 0.62 * g[..., 1], 0, None)
        g[..., 0] = np.clip(g[..., 0] - 1.00 * excess_b, 0, 255)
        g[..., 1] = np.clip(g[..., 1] * 1.04 + 3.0, 0, 255)
        g[..., 2] = np.clip(g[..., 2] * 1.10 + 8.0, 0, 255)
        graded = graded * (1.0 - 0.90 * op[..., None]) + g * (0.90 * op[..., None])
    sub = sub_q * (1.0 - m[..., None]) + graded * m[..., None]
    out = imgf.copy()
    out[y0:y1, x0:x1] = sub
    return np.clip(out, 0, 255)


def process_one(src_q, src_v, cat_a):
    img = remove_dust(src_q, cat_a)
    img = restore_ears(img, src_v, cat_a)
    img = restore_mouth(img, src_v)
    img = np.clip(img, 0, 255).astype(np.uint8)
    y0, y1, x0, x1 = TITLE
    img = img.copy()
    img[y0:y1, x0:x1] = src_q[y0:y1, x0:x1]
    return img


def process(src_q, src_v, mattes, preview=None):
    want = set(preview) if preview else set(range(NFR))
    out = [None] * NFR
    for i in range(NFR):
        if i not in want:
            continue
        cat = mattes[i]
        out[i] = process_one(src_q[i], src_v[i], cat)
        if preview or i % 25 == 0:
            Yi = luma(out[i])
            print(f'  n{i} glob={Yi.mean():.1f} bulb={Yi[155:185, 300:330].max():.0f}',
                  flush=True)
    return out


def save_preview(out, src_q, src_v, idx, dest):
    dest.mkdir(parents=True, exist_ok=True)
    crops = {
        'head': (160, 460, 570, 950),
        'mouth': (360, 460, 640, 820),
        'lear': (130, 230, 520, 720),
        'rear': (160, 280, 800, 980),
        'crown': (125, 200, 450, 820),
        'steam': (250, 520, 300, 700),
        'lamp': (80, 220, 200, 460),
        'window': (0, 420, 250, 860),
        'tail': (480, 720, 880, 1280),
        'screen': (360, 580, 0, 230),
    }
    thumbs = []
    for i in idx:
        if out[i] is None:
            continue
        cv2.imwrite(str(dest / f'full-n{i}.png'), out[i])
        for tag, (ya, yb, xa, xb) in crops.items():
            z = cv2.resize(out[i][ya:yb, xa:xb], None, fx=3, fy=3,
                           interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(str(dest / f'{tag}-3x-n{i}.jpg'), z,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        t = cv2.resize(out[i], (640, 360))
        cv2.putText(t, f'n{i}', (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (240, 240, 240), 2)
        thumbs.append(t)
    if thumbs:
        while len(thumbs) % 4:
            thumbs.append(np.zeros((360, 640, 3), np.uint8))
        grid = np.vstack([np.hstack(thumbs[r * 4:(r + 1) * 4]) for r in range(len(thumbs) // 4)])
        cv2.imwrite(str(dest / 'contact-sheet.jpg'), grid, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    # side-by-side v41 | q-ex | r-ex at n138, n370, n432
    rows = []
    for i in (138, 370, 432):
        if out[i] is None or src_v is None:
            continue
        cells = []
        for im, lab in ((src_v[i], 'v41'), (src_q[i], 'v41q-ex'), (out[i], 'v41r-ex')):
            c = cv2.resize(im[160:460, 570:950], None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
            cv2.putText(c, f'{lab} n{i}', (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cells.append(c)
        rows.append(np.hstack(cells))
    if rows:
        sheet = np.vstack(rows)
        dest2 = P / 'qc' / VER
        dest2.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dest2 / 'head-compare-preview.png'), sheet)
        cv2.imwrite(str(dest / 'head-compare-preview.png'), sheet)
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
        ('v41-los', SRC_V, V41_LOS_SHA),
        ('v41q-ex-mp4', P / 'v41q-ex-test-25s.mp4', V41Q_SHA),
        ('v41q-ex-los', SRC_Q, V41Q_LOS_SHA),
        ('pcm', PCM, Hh.PCM_SHA),
    ):
        got = sha256(path)
        if got != expect:
            raise SystemExit(f'{label} sha changed: {got}')
    print('frozen shas ok', flush=True)

    print('loading v41q-ex and v41 lossless…', flush=True)
    src_q = Hh.load_span(SRC_Q)
    src_v = Hh.load_span(SRC_V)
    assert len(src_q) == NFR and len(src_v) == NFR
    print('loading cat mattes…', flush=True)
    mattes = Hh.load_span(MATTE, gray=True)

    if args.preview:
        idx = [0, 6, 88, 89, 100, 120, 125, 130, 135, 138, 140, 145, 150, 155, 160,
               241, 300, 350, 360, 368, 370, 372, 380, 432]
        out = process(src_q, src_v, mattes, preview=idx)
        save_preview(out, src_q, src_v, idx, QC / 'preview')
        save_preview(out, src_q, src_v, idx, WORK / 'preview')
        return

    out = process(src_q, src_v, mattes)
    y0, y1, x0, x1 = TITLE
    bad = [i for i, (a, b) in enumerate(zip(src_q, out))
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
    shutil.copy2(Path(__file__), WORK / 'build_v41r_ex.py')
    print('v41 still', sha256(Hh.V41_MP4), flush=True)
    print('v41q-ex still', sha256(P / 'v41q-ex-test-25s.mp4'), flush=True)


if __name__ == '__main__':
    main()
