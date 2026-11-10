#!/usr/bin/env python3
"""v41y-ex: composite generated cat + plant onto frozen u-ex plates.

gen1 (gentle meow, organic tail) covers the n138 cycle; gen2 (full meow
with tongue and fangs) covers the n432 cycle. Both clips start and end
on u-ex n0, so cycle joins stay on the same rest pose. Only the cat
(head, body, tail) and the right-corner plant are taken from the
generated clips; snow, steam, window, lamp, desk, title and audio stay
on the u-ex plates.

Does not edit earlier builders or encodes.
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
import build_v41t_ex as T

VER = 'v41y-ex'
WORK = P / VER
QC = P / 'qc' / VER
SRC_U = P / 'v41u-ex' / 'v41u-ex-span-890-915-lossless.mkv'
GEN1 = P / 'genvid' / 'gen1.mp4'
GEN2 = P / 'genvid' / 'gen2.mp4'
RAW1 = WORK / 'mattes' / 'gen1-raw-u2net.npz'
RAW2 = WORK / 'mattes' / 'gen2-raw-u2net.npz'
PCM = Hh.PCM

V41_SHA = Hh.V41_SHA
V41U_SHA = 'e14aa352424f4cc729c254f255b04907cc770063216083e8e970be1858e92560'
V41U_LOS_SHA = '018e7829b02b3eb5343a3b5bd243fae2077306bbd9241bfd1d465995276e5dc5'

NFR, W, H, NCYCLE = 600, 1280, 720, 241
TITLE = Hh.TITLE
Yg, Xg = np.mgrid[0:H, 0:W].astype(np.float32)

# Visual peaks in the generated clips, mapped onto the audio peaks.
# gen1's tongue-out peak is ~n132; audio medium peak is n138.
# gen2's wide open plateau is ~n195-210 (n200 has both eyes open);
# n432 is local index 191 of the second 241-cycle (432 % 241 == 191).
GEN1_PEAK, DST_A = 132.0, 138.0
GEN2_PEAK, DST_B = 200.0, 191.0

sha256 = Hh.sha256 if hasattr(Hh, 'sha256') else None


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
    return T.sstep(x)


def load_mp4(path):
    cap = cv2.VideoCapture(str(path))
    fr = []
    while True:
        ok, im = cap.read()
        if not ok:
            break
        fr.append(im)
    cap.release()
    return fr


def src_time(local, src_pk, dst_pk, last=240):
    """Piecewise-linear time map: pin 0 and last, move src_pk to dst_pk."""
    local = np.clip(float(local), 0.0, float(last))
    if local <= dst_pk:
        return local * (src_pk / max(dst_pk, 1e-6))
    return src_pk + (local - dst_pk) * ((last - src_pk) / max(last - dst_pk, 1e-6))


def lerp_frame(frames, t):
    n = len(frames)
    if t <= 0:
        return frames[0]
    if t >= n - 1:
        return frames[-1]
    i0 = int(np.floor(t))
    u = float(t - i0)
    a = frames[i0].astype(np.float32)
    b = frames[i0 + 1].astype(np.float32)
    return np.clip(a * (1.0 - u) + b * u, 0, 255)


def lerp_alpha(stack, t):
    n = len(stack)
    if t <= 0:
        return stack[0].astype(np.float32)
    if t >= n - 1:
        return stack[-1].astype(np.float32)
    i0 = int(np.floor(t))
    u = float(t - i0)
    return np.clip(stack[i0] * (1.0 - u) + stack[i0 + 1] * u, 0, 1)


def fill_interior(a, thresh=0.40):
    """Fill holes (pupils, mouth, nostrils) so u-ex does not show through."""
    hard = (a >= thresh).astype(np.uint8)
    cnts, _ = cv2.findContours(hard, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(hard)
    cv2.drawContours(filled, cnts, -1, 1, thickness=-1)
    interior = cv2.erode(filled, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    out = np.where(interior > 0, np.maximum(a, 0.99), a)
    return np.clip(out.astype(np.float32), 0, 1)


def plant_mask(im):
    imf = im.astype(np.float32)
    B, G, R = imf[..., 0], imf[..., 1], imf[..., 2]
    Y = luma(imf)
    roi = (Xg > 970) & (Yg > 460)
    leaf = roi & (G > R + 2) & (G > B - 10) & (Y < 145) & (Y > 10)
    olive = roi & (G > R - 4) & (G > B) & (Y < 120) & (Y > 16) & ((G - R) < 45)
    m = (leaf | olive).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    keep = np.zeros_like(m)
    areas = sorted(((st[i, cv2.CC_STAT_AREA], i) for i in range(1, n)), reverse=True)
    for area, i in areas[:6]:
        if area < 60:
            continue
        keep[lab == i] = 1
    # don't let the plant matte climb onto the brown chair or the mic
    chair = roi & (R > G + 8) & (R > B + 8) & (Y > 25) & (Y < 90)
    keep[chair] = 0
    a = cv2.GaussianBlur(keep.astype(np.float32), (0, 0), 1.8)
    return np.clip(a, 0, 1)


def clean_cat(im, raw, pane, pane_allow, ceil=None):
    """Punch lamp, title and window glass (except the ear prior); keep one cat."""
    a = raw.astype(np.float32).copy()
    if a.max() > 1.5:
        a = a / 255.0
    a[0:250, 0:480] = 0
    y0, y1, x0, x1 = TITLE
    a[y0:y1, x0:x1] = 0
    punch = pane * (1.0 - pane_allow)
    a = a * (1.0 - punch)
    # sky / night leftover in the pane
    Y = luma(im)
    B = im[..., 0].astype(np.float32)
    R = im[..., 2].astype(np.float32)
    sky = ((pane > 0.5) & (Y < 100) & (B + 8 >= R) & (pane_allow < 0.5)).astype(np.float32)
    a = a * (1.0 - sky)
    bin_ = (a > 0.35).astype(np.uint8)
    n, lab, st, cents = cv2.connectedComponentsWithStats(bin_, 8)
    best, best_area = 0, 0
    for i in range(1, n):
        area = st[i, cv2.CC_STAT_AREA]
        cx, cy = cents[i]
        if area < 2000 or cx < 400 or cy < 140:
            continue
        if area > best_area:
            best, best_area = i, area
    if best == 0 and n > 1:
        best = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    keep = (lab == best).astype(np.uint8)
    dil = cv2.dilate(keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    out = (raw.astype(np.float32) / (255.0 if raw.max() > 1.5 else 1.0)) * dil.astype(np.float32)
    out[0:250, 0:480] = 0
    out = out * (1.0 - punch) * (1.0 - sky)
    out[y0:y1, x0:x1] = 0
    if ceil is not None:
        out = apply_ceiling(out, ceil)
    out = fill_interior(out)
    if ceil is not None:
        out = apply_ceiling(out, ceil)
    return np.clip(out, 0, 1)


def temporal_gauss(stack, sig=1.0):
    r = int(3 * sig) + 1
    xs = np.arange(-r, r + 1, dtype=np.float32)
    w = np.exp(-0.5 * (xs / sig) ** 2)
    w = w / w.sum()
    out = np.zeros_like(stack, np.float32)
    for k, wk in zip(range(-r, r + 1), w):
        out += wk * np.roll(stack, k, axis=0)
    return np.clip(out, 0, 1)


def build_mattes(frames, raw, pane, pane_allow, ceil=None):
    """Per-frame cat + held-stable plant, temporally smoothed, contact edges held."""
    Tlen = len(frames)
    cat_s = np.zeros((Tlen, H, W), np.float32)
    plant_held = held_plant(frames)
    print('    held plant area', int((plant_held > 0.5).sum()), flush=True)
    for i in range(Tlen):
        raw_i = raw[i]
        cat_s[i] = clean_cat(frames[i], raw_i, pane, pane_allow, ceil=ceil)
        if i % 40 == 0:
            print(f'    post {i}/{Tlen}', flush=True)
    cat_s = temporal_gauss(cat_s, sig=0.85)
    if ceil is not None:
        for i in range(Tlen):
            cat_s[i] = apply_ceiling(cat_s[i], ceil)
    plant_s = np.broadcast_to(plant_held, cat_s.shape).copy()
    # Held-stable contact bands (window + desk) to stop 1px edge crawl.
    long = temporal_gauss(cat_s, sig=3.2)
    desk = ((Yg > 560) & (Xg > 480) & (Xg < 960)).astype(np.float32)
    # a band around the pane / frame where the ear and cheek meet the window
    pane_band = cv2.dilate((pane > 0.5).astype(np.uint8),
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))).astype(np.float32)
    # only the cat's silhouette edge in those regions
    contact = np.clip(np.maximum(desk, pane_band * sstep((Xg - 480.0) / 8.0)), 0, 1)
    contact = cv2.GaussianBlur(contact, (0, 0), 4.0)
    for i in range(Tlen):
        edge = (cat_s[i] > 0.08) & (cat_s[i] < 0.92)
        w = contact * edge.astype(np.float32)
        cat_s[i] = cat_s[i] * (1.0 - w) + long[i] * w
    a = np.clip(np.maximum(cat_s, plant_s), 0, 1)
    # hard-zero dust so the background stays byte-identical
    a[a < 0.03] = 0
    return a, cat_s, plant_s


def pane_allow_from(n0_cat, pane):
    """Only the n0 ear/head may live in front of the glass; never the sky."""
    prior = cv2.dilate((n0_cat > 0.4).astype(np.uint8),
                       cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))).astype(np.float32)
    # also require it sit on the right side of the left pane / left of right pane
    # (the ear zone), not a blob in mid-glass
    return np.clip(prior * pane, 0, 1)


def head_ceiling(n0_cat, margin=40):
    """Per-column y above which nothing is allowed (kills window-frame spikes)."""
    hard = cv2.dilate((n0_cat > 0.45).astype(np.uint8),
                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 9)))
    ceil = np.full(W, H, np.float32)
    for x in range(W):
        ys = np.flatnonzero(hard[:, x])
        if ys.size:
            ceil[x] = max(0, int(ys[0]) - margin)
    ceil = cv2.blur(ceil.reshape(1, -1), (41, 1)).reshape(-1)
    return np.clip(ceil, 0, H)


def apply_ceiling(a, ceil, x0=480, x1=980):
    kill = (Xg >= x0) & (Xg < x1) & (Yg < ceil[None, :])
    out = a.copy()
    out[kill] = 0
    return out


def held_plant(frames):
    """Stable plant region: confident leaf pixels, holes filled, held for the cycle."""
    acc = np.zeros((H, W), np.float32)
    p0 = plant_mask(frames[0])
    p1 = plant_mask(frames[-1])
    for im in frames:
        acc += (plant_mask(im) > 0.45).astype(np.float32)
    freq = acc / float(len(frames))
    core = ((freq > 0.20) | (p0 > 0.40) | (p1 > 0.40)).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    core = cv2.morphologyEx(core, cv2.MORPH_CLOSE, k)
    cnts, _ = cv2.findContours(core, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(core)
    cv2.drawContours(filled, cnts, -1, 1, thickness=-1)
    filled = cv2.dilate(filled, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
    # re-punch the brown chair / mic so dilation doesn't climb onto them
    im0 = frames[0].astype(np.float32)
    B, G, R = im0[..., 0], im0[..., 1], im0[..., 2]
    Y = 0.114 * B + 0.587 * G + 0.299 * R
    roi = (Xg > 970) & (Yg > 460)
    chair = roi & (R > G + 8) & (R > B + 8) & (Y > 25) & (Y < 90)
    filled[chair.astype(bool)] = 0
    filled[Xg <= 970] = 0
    filled[Yg <= 460] = 0
    a = cv2.GaussianBlur(filled.astype(np.float32), (0, 0), 1.6)
    return np.clip(a, 0, 1)


def color_match(gen, uex, alpha):
    """Match dusk grade and lighting of the composited cat to this u-ex frame."""
    m = ((alpha > 0.70) & (Yg > 360) & (Yg < 560) & (Xg > 560) & (Xg < 900))
    Yg_ = luma(gen)
    Yu = luma(uex)
    m = m & (Yg_ > 70) & (Yg_ < 200) & (Yu > 70) & (Yu < 200)
    if int(m.sum()) < 80:
        return gen
    gs = cv2.cvtColor(np.clip(gen, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    ds = cv2.cvtColor(np.clip(uex, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    out = gs.copy()
    for c in range(3):
        mu_s, sd_s = float(gs[..., c][m].mean()), float(gs[..., c][m].std()) + 1e-6
        mu_d, sd_d = float(ds[..., c][m].mean()), float(ds[..., c][m].std()) + 1e-6
        out[..., c] = (gs[..., c] - mu_s) * (0.65 * sd_d / sd_s + 0.35) + mu_d
    rgb = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)
    # mild grain from the destination
    grain = uex.astype(np.float32) - cv2.GaussianBlur(uex.astype(np.float32), (0, 0), 1.15)
    rgb = rgb + 0.45 * grain * alpha[..., None]
    return np.clip(rgb, 0, 255)


def residual_flicker(img, w_room, w_bulb, ph, ph0):
    sig = T.flicker_signal(ph) - T.flicker_signal(ph0)
    if abs(sig) < 1e-4:
        return img
    warm = np.array([0.72, 0.92, 1.06], np.float32)
    fk = (0.022 * sig * w_room + 0.013 * sig * w_bulb)
    return np.clip(img.astype(np.float32) * (1.0 + fk[..., None] * warm), 0, 255)


def cycle_of(i):
    """Which generated clip and local index for output frame i."""
    local = i % NCYCLE
    # first and third cycles: gen1 (gentle meow at local 138)
    # second cycle (241-481): gen2 (full meow at local 191 == n432)
    use_gen2 = (241 <= i <= 481)
    return use_gen2, local


def compose_one(uex, gen, alpha, ph, ph0, w_room, w_bulb):
    gen_f = residual_flicker(gen, w_room, w_bulb, ph, ph0)
    gen_f = color_match(gen_f, uex, alpha)
    a = alpha.astype(np.float32)
    a = np.clip(a, 0, 1)
    a = a * a  # slightly tighter edge (less halo)
    # actually squaring darkens the fur fringe too much; use a soft knee
    # revert to linear with a small threshold already applied
    a = alpha.astype(np.float32)
    out = uex.astype(np.float32) * (1.0 - a[..., None]) + gen_f * a[..., None]
    # exact u-ex where the matte is off
    off = a < 0.02
    out[off] = uex[off].astype(np.float32)
    y0, y1, x0, x1 = TITLE
    out[y0:y1, x0:x1] = uex[y0:y1, x0:x1]
    return np.clip(out, 0, 255).astype(np.uint8)


def process(src_u, g1, g2, a1, a2, w_room, w_bulb, preview=None):
    want = set(preview) if preview else set(range(NFR))
    out = [None] * NFR
    ph0 = Hh.phase_of(0)
    for i in range(NFR):
        if i not in want:
            continue
        use2, local = cycle_of(i)
        if use2:
            t = src_time(local, GEN2_PEAK, DST_B)
            gen = lerp_frame(g2, t)
            al = lerp_alpha(a2, t)
        else:
            t = src_time(local, GEN1_PEAK, DST_A)
            gen = lerp_frame(g1, t)
            al = lerp_alpha(a1, t)
        ph = Hh.phase_of(i)
        out[i] = compose_one(src_u[i], gen, al, ph, ph0, w_room, w_bulb)
        if preview or i % 25 == 0:
            print(f'  n{i} local={local} gen{"2" if use2 else "1"} t={t:.2f}', flush=True)
    return out


# ---------------- QC ----------------
def face_grid(out, dest, every=10):
    dest.mkdir(parents=True, exist_ok=True)
    tiles = []
    for i in range(0, 241, every):
        if out[i] is None:
            continue
        c = cv2.resize(out[i][180:430, 520:900], (240, 160), interpolation=cv2.INTER_AREA)
        cv2.putText(c, f'n{i}', (6, 18), cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 255, 255), 1)
        tiles.append(c)
    if not tiles:
        return
    cols = 5
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[k:k + cols]) for k in range(0, len(tiles), cols)]
    path = dest / 'face-grid-every-10.png'
    cv2.imwrite(str(path), np.vstack(rows))
    print('face grid', path)


def meow_seq(out, dest):
    dest.mkdir(parents=True, exist_ok=True)
    for tag, rng in (('138', range(126, 147, 2)), ('432', range(420, 445, 2))):
        cells = []
        for i in rng:
            if out[i] is None:
                continue
            c = cv2.resize(out[i][180:460, 520:900], None, fx=3, fy=3,
                           interpolation=cv2.INTER_NEAREST)
            cv2.putText(c, f'n{i}', (8, 28), cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255), 2)
            cells.append(c)
        if cells:
            path = dest / f'meow-seq-{tag}.png'
            cv2.imwrite(str(path), np.hstack(cells))
            print('meow seq', path)


def tail_strip(out, dest, every=6):
    dest.mkdir(parents=True, exist_ok=True)
    cells = []
    for i in range(0, 241, every):
        if out[i] is None:
            continue
        c = out[i][400:720, 880:1280]
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


def before_after(out, src_u, dest):
    dest.mkdir(parents=True, exist_ok=True)
    mouths = []
    for i in (138, 432):
        if out[i] is None:
            continue
        a = cv2.resize(src_u[i][180:460, 520:900], None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
        b = cv2.resize(out[i][180:460, 520:900], None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
        pair = np.hstack([a, b])
        cv2.putText(pair, f'u n{i}', (8, 28), cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255), 2)
        cv2.putText(pair, f'{VER} n{i}', (a.shape[1] + 8, 28), cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255), 2)
        mouths.append(pair)
    if mouths:
        path = dest / 'mouth-u-vs-new-n138-n432.png'
        cv2.imwrite(str(path), np.vstack(mouths))
        print('mouth ba', path)
    tails = []
    for i in (0, 120, 180, 432):
        if out[i] is None:
            continue
        a = src_u[i][400:720, 880:1280]
        b = out[i][400:720, 880:1280]
        pair = np.hstack([a, b])
        cv2.putText(pair, f'u n{i}', (8, 24), cv2.FONT_HERSHEY_PLAIN, 1.6, (255, 255, 255), 2)
        cv2.putText(pair, f'{VER} n{i}', (a.shape[1] + 8, 24), cv2.FONT_HERSHEY_PLAIN, 1.6, (255, 255, 255), 2)
        tails.append(pair)
    if tails:
        path = dest / 'tail-u-vs-new.png'
        cv2.imwrite(str(path), np.vstack(tails))
        print('tail ba', path)


def seam_qc(out, src_u, dest):
    dest.mkdir(parents=True, exist_ok=True)
    for i, label in ((0, 'n0'), (240, 'n240'), (241, 'n241'), (481, 'n481'), (482, 'n482')):
        if i >= len(out) or out[i] is None:
            continue
        # full frame small + face + tail
        full = cv2.resize(out[i], (640, 360), interpolation=cv2.INTER_AREA)
        cv2.putText(full, label, (8, 24), cv2.FONT_HERSHEY_PLAIN, 1.6, (255, 255, 255), 2)
        cv2.imwrite(str(dest / f'seam-{label}.png'), full)
    # side by side n240 vs n241 vs n0
    if out[0] is not None and out[240] is not None and out[241] is not None:
        def face(im):
            return cv2.resize(im[180:430, 520:900], (400, 260), interpolation=cv2.INTER_AREA)
        row = np.hstack([face(out[0]), face(out[240]), face(out[241])])
        cv2.putText(row, 'n0', (8, 24), cv2.FONT_HERSHEY_PLAIN, 1.6, (255, 255, 255), 2)
        cv2.putText(row, 'n240', (408, 24), cv2.FONT_HERSHEY_PLAIN, 1.6, (255, 255, 255), 2)
        cv2.putText(row, 'n241', (808, 24), cv2.FONT_HERSHEY_PLAIN, 1.6, (255, 255, 255), 2)
        cv2.imwrite(str(dest / 'seam-faces-n0-n240-n241.png'), row)
        if out[481] is not None and out[482] is not None:
            row2 = np.hstack([face(out[481]), face(out[482]), face(out[0])])
            cv2.putText(row2, 'n481', (8, 24), cv2.FONT_HERSHEY_PLAIN, 1.6, (255, 255, 255), 2)
            cv2.putText(row2, 'n482', (408, 24), cv2.FONT_HERSHEY_PLAIN, 1.6, (255, 255, 255), 2)
            cv2.putText(row2, 'n0', (808, 24), cv2.FONT_HERSHEY_PLAIN, 1.6, (255, 255, 255), 2)
            cv2.imwrite(str(dest / 'seam-faces-n481-n482-n0.png'), row2)
    print('seams written', dest)


def edge_and_bg_qc(out, src_u, dest):
    dest.mkdir(parents=True, exist_ok=True)
    # window / desk edges and plant at rest + meows
    for i, tag in ((0, 'n0'), (138, 'n138'), (241, 'n241'), (432, 'n432')):
        if out[i] is None:
            continue
        def trip(box, name, sc=2):
            y0, y1, x0, x1 = box
            p = np.hstack([src_u[i][y0:y1, x0:x1], out[i][y0:y1, x0:x1]])
            p = cv2.resize(p, None, fx=sc, fy=sc, interpolation=cv2.INTER_NEAREST)
            cv2.putText(p, f'u {name} {tag}', (8, 24), cv2.FONT_HERSHEY_PLAIN, 1.6, (255, 255, 255), 2)
            cv2.imwrite(str(dest / f'{name}-{tag}.png'), p)
        trip((80, 360, 480, 760), 'window-edge')
        trip((500, 720, 480, 900), 'desk-edge')
        trip((480, 720, 960, 1280), 'plant')
    # background MAE where we expect u-ex to be untouched: window glass
    # (minus a dilated cat) and left desk
    pane = T.pane_masks().astype(np.float32)
    rows = []
    for i in range(0, 600, 10):
        if out[i] is None:
            continue
        d = cv2.absdiff(out[i], src_u[i]).astype(np.float32)
        # title should be identical
        y0, y1, x0, x1 = TITLE
        title_m = float(d[y0:y1, x0:x1].mean())
        # left desk / mug / steam (no cat)
        desk = d[400:720, 80:500]
        rows.append((i, title_m, float(desk.mean()), float(d.mean())))
    path = dest / 'bg-mae.txt'
    with open(path, 'w') as f:
        f.write('i title_mae left_desk_mae global_mae\n')
        for r in rows:
            f.write(f'{r[0]:4d}  {r[1]:.4f}  {r[2]:.4f}  {r[3]:.4f}\n')
    print('bg mae', path, 'n samples', len(rows),
          'title mean', np.mean([r[1] for r in rows]) if rows else None,
          'desk mean', np.mean([r[2] for r in rows]) if rows else None)


def eye_report(out, src_u, dest):
    """Rough eye width/height of u-ex vs y at rest and both meows."""
    dest.mkdir(parents=True, exist_ok=True)
    # crop both eyes, threshold dark pupil/iris
    boxes = {'L': (230, 320, 600, 710), 'R': (230, 320, 740, 860)}
    lines = []
    for i, tag in ((0, 'n0'), (138, 'n138'), (432, 'n432')):
        for side, (y0, y1, x0, x1) in boxes.items():
            for name, im in (('u', src_u[i]), ('y', out[i] if out[i] is not None else src_u[i])):
                if im is None:
                    continue
                sub = im[y0:y1, x0:x1]
                Y = luma(sub)
                dark = (Y < 70).astype(np.uint8)
                dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
                n, lab, st, _ = cv2.connectedComponentsWithStats(dark, 8)
                if n <= 1:
                    lines.append(f'{tag} {name} {side} no-blob')
                    continue
                j = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
                w, h = st[j, cv2.CC_STAT_WIDTH], st[j, cv2.CC_STAT_HEIGHT]
                lines.append(f'{tag} {name} {side} {w}x{h} area={st[j, cv2.CC_STAT_AREA]}')
    text = '\n'.join(lines)
    (dest / 'eye-sizes.txt').write_text(text + '\n')
    print('eyes\n' + text)
    # visual compare n0 / n138 / n432 eyes 3x
    for i, tag in ((0, 'n0'), (138, 'n138'), (432, 'n432')):
        if out[i] is None:
            continue
        a = cv2.resize(src_u[i][210:340, 590, 880] if False else src_u[i][210:340, 590:880],
                       None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
        b = cv2.resize(out[i][210:340, 590:880], None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
        pair = np.hstack([a, b])
        cv2.putText(pair, f'u {tag}', (8, 24), cv2.FONT_HERSHEY_PLAIN, 1.6, (255, 255, 255), 2)
        cv2.putText(pair, f'y {tag}', (a.shape[1] + 8, 24), cv2.FONT_HERSHEY_PLAIN, 1.6, (255, 255, 255), 2)
        cv2.imwrite(str(dest / f'eyes-u-vs-y-{tag}.png'), pair)


def encoded_stills(mp4, dest):
    dest.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(mp4))
    want = {0, 120, 138, 180, 240, 241, 432, 481, 482}
    frames = {}
    i = 0
    while True:
        ok, im = cap.read()
        if not ok:
            break
        if i in want:
            frames[i] = im
        i += 1
    cap.release()
    jobs = [
        (138, (180, 460, 520, 900), 'encoded-mouth-n138', 3),
        (432, (180, 460, 520, 900), 'encoded-mouth-n432', 3),
        (0, (400, 720, 880, 1280), 'encoded-tail-n0', 2),
        (120, (400, 720, 880, 1280), 'encoded-tail-n120', 2),
        (180, (400, 720, 880, 1280), 'encoded-tail-n180', 2),
        (432, (400, 720, 880, 1280), 'encoded-tail-n432', 2),
        (138, (80, 360, 480, 760), 'encoded-window-n138', 2),
        (432, (80, 360, 480, 760), 'encoded-window-n432', 2),
        (0, (500, 720, 480, 900), 'encoded-desk-n0', 2),
        (138, (200, 360, 580, 900), 'encoded-face-n138', 2),
        (432, (200, 360, 580, 900), 'encoded-face-n432', 2),
    ]
    for i, box, name, sc in jobs:
        if i not in frames:
            continue
        y0, y1, x0, x1 = box
        z = cv2.resize(frames[i][y0:y1, x0:x1], None, fx=sc, fy=sc,
                       interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(dest / f'{name}.png'), z)
    for i in want:
        if i in frames:
            cv2.imwrite(str(dest / f'encoded-n{i}.png'), frames[i])
    return frames


def write_qc(out, src_u, dest, mp4=None):
    dest.mkdir(parents=True, exist_ok=True)
    face_grid(out, dest)
    meow_seq(out, dest)
    tail_strip(out, dest)
    before_after(out, src_u, dest)
    seam_qc(out, src_u, dest)
    edge_and_bg_qc(out, src_u, dest)
    eye_report(out, src_u, dest)
    if mp4 is not None:
        encoded_stills(mp4, dest)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--preview', action='store_true')
    ap.add_argument('--use-cached-mattes', action='store_true')
    args = ap.parse_args()

    mp4 = P / f'{VER}-test-25s.mp4'
    if not args.preview and mp4.exists():
        raise SystemExit(f'refuse to overwrite {mp4}')

    for label, path, expect in (
        ('v41u-ex-mp4', P / 'v41u-ex-test-25s.mp4', V41U_SHA),
        ('v41u-ex-los', SRC_U, V41U_LOS_SHA),
        ('pcm', PCM, Hh.PCM_SHA),
    ):
        got = sha256(path)
        if got != expect:
            raise SystemExit(f'{label} sha changed: {got}')
    print('frozen shas ok', flush=True)

    if not RAW1.exists() or not RAW2.exists():
        raise SystemExit(f'missing u2net cache {RAW1} / {RAW2}')

    print('loading plates and generated clips…', flush=True)
    src_u = Hh.load_span(SRC_U)
    g1 = load_mp4(GEN1)
    g2 = load_mp4(GEN2)
    assert len(src_u) == NFR and len(g1) == NCYCLE and len(g2) == NCYCLE
    raw1 = np.load(RAW1)['alpha']
    raw2 = np.load(RAW2)['alpha']
    assert raw1.shape[0] == NCYCLE and raw2.shape[0] == NCYCLE

    FIN1 = WORK / 'mattes' / 'gen1-final.npz'
    FIN2 = WORK / 'mattes' / 'gen2-final.npz'
    pane = T.pane_masks().astype(np.float32)
    if args.use_cached_mattes and FIN1.exists() and FIN2.exists():
        print('loading cached final mattes…', flush=True)
        a1 = np.load(FIN1)['alpha'].astype(np.float32) / 255.0
        a2 = np.load(FIN2)['alpha'].astype(np.float32) / 255.0
    else:
        print('building mattes…', flush=True)
        # The face sits in front of PANE_R. Build the n0 prior WITHOUT punching
        # the pane, otherwise the prior is empty on the face and later frames
        # leave a rectangular hole there.
        allow_all = np.ones((H, W), np.float32)
        n0_cat = clean_cat(g1[0], raw1[0], pane, allow_all)
        pane_allow = pane_allow_from(n0_cat, pane)
        # grow enough to cover a meow (head drops / cheeks puff a little)
        pane_allow = np.clip(cv2.dilate((pane_allow > 0.2).astype(np.uint8),
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))).astype(np.float32), 0, 1)
        pane_allow = pane_allow * pane
        # never allow the window-frame / hanging-vine spikes above the resting head
        ceil = head_ceiling(n0_cat, margin=40)
        # also shrink pane_allow so it cannot climb the glass above the head
        pane_allow = apply_ceiling(pane_allow, ceil)
        print('  head ceiling mid-face', float(ceil[700]), 'mullion', float(ceil[580]), flush=True)
        print('  gen1 mattes', flush=True)
        a1, _, _ = build_mattes(g1, raw1, pane, pane_allow, ceil=ceil)
        print('  gen2 mattes', flush=True)
        a2, _, _ = build_mattes(g2, raw2, pane, pane_allow, ceil=ceil)
        np.savez_compressed(FIN1, alpha=(a1 * 255).astype(np.uint8))
        np.savez_compressed(FIN2, alpha=(a2 * 255).astype(np.uint8))

    pane_m = T.pane_masks().astype(np.float32)
    # lamp weights from a union of rest mattes (stable)
    cat_union = np.clip(np.maximum(a1[0], a2[0]), 0, 1)
    w_room, w_bulb = T.lamp_weights(pane_m, cat_union)
    QC.mkdir(parents=True, exist_ok=True)

    if args.preview:
        idx = sorted(set(
            list(range(0, 241, 10))
            + list(range(126, 147))
            + list(range(230, 252))
            + list(range(420, 445))
            + list(range(470, 500))
            + [599]
        ))
        print('preview frames', len(idx), flush=True)
        out = process(src_u, g1, g2, a1, a2, w_room, w_bulb, preview=idx)
        dest = QC / 'preview'
        write_qc(out, src_u, dest, mp4=None)
        print('preview written', dest, flush=True)
        return

    out = process(src_u, g1, g2, a1, a2, w_room, w_bulb)
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
    shutil.copy2(Path(__file__), WORK / 'build_v41y_ex.py')
    dest = QC / 'encoded'
    write_qc(out, src_u, dest, mp4=mp4)


if __name__ == '__main__':
    main()
