#!/usr/bin/env python3
"""v41x-ex: strategic mouth + tail redo on frozen v41u-ex.

Mouth: transfer a real greenscreen meow (ref2 n245 medium / n250 big)
onto the muzzle with a scale+translate (similarity, no forced twist),
Reinhard/painterly match, tight oral feather. Jaw/chin drop with dest
fur. Eyes and ears stay locked to u-ex.

Tail: inverse-map the painted o-ex sprite along the 20-bone chain, but
with a soft fur-edge alpha (no hard matte), the root emerging behind a
furry haunch occlusion (no straight cut), and a full static leaf plate
wherever the old or new tail is absent.

Does not edit earlier builders or encodes. Audio is the approved v41 PCM.
Title is copied through.
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
import build_v41w_ex as Wx

VER = 'v41x-ex'
WORK = P / VER
QC = P / 'qc' / VER
SRC_U = P / 'v41u-ex' / 'v41u-ex-span-890-915-lossless.mkv'
SRC_Q = P / 'v41q-ex' / 'v41q-ex-span-890-915-lossless.mkv'
MATTE = Hh.MATTE
PCM = Hh.PCM
REF2 = Path('/workspace/ref-greenscreen/greenscreen-meow-ref2.mp4')

V41_SHA = Hh.V41_SHA
V41Q_SHA = 'be65528916d68b99cc64546d865388f824656b3975adb8f861f10d03e548afcd'
V41U_SHA = 'e14aa352424f4cc729c254f255b04907cc770063216083e8e970be1858e92560'
V41U_LOS_SHA = '018e7829b02b3eb5343a3b5bd243fae2077306bbd9241bfd1d465995276e5dc5'
V41W_SHA = '40088c8bc082010880a95c9bc8542c1692b2dcbf978290668d564df1b0acd583'

NFR, Wd, Ht, NCYCLE = 600, 1280, 720, 241
W, H = Wd, Ht
TITLE = Hh.TITLE
Yg, Xg = np.mgrid[0:H, 0:W].astype(np.float32)

# Dest / ref landmarks (hand-checked on 3x crops)
U_NOSE = np.array([716.0, 364.0], np.float32)
# Visual centre of the u-ex slit (the loose Y<95 detector was centred on
# the search box, not the opening — the cavity sits left of the nose).
U_MOUTH_A = np.array([662.0, 396.0], np.float32)
U_MOUTH_B = np.array([692.0, 404.0], np.float32)
# ref2 frames, nose on the leather (not the philtrum), mouth on the opening
REF_LM = {
    245: (np.array([328.0, 124.0], np.float32), np.array([338.0, 192.0], np.float32)),
    250: (np.array([328.0, 122.0], np.float32), np.array([339.0, 198.0], np.float32)),
}

sha256 = Wx.sha256
luma = Wx.luma
sstep = T.sstep


# ---------------- leaf plate + tail ----------------
def build_full_leaf_plate(src_q, mattes, seed_bg):
    """Static plant fill over the whole hook/plant corner, not just x>1088.

    Median of leafy q-ex samples. Remaining pale-fur holes are filled from
    the nearest leafy pixel (no broad Telea fog).
    """
    bg = seed_bg.astype(np.float32).copy()
    acc = np.zeros((H, W, 3), np.float64)
    wsum = np.zeros((H, W), np.float64)
    roi = (Xg > 980) & (Yg > 510)
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
    have = wsum > 4.0
    bg[have] = (acc[have] / wsum[have, None]).astype(np.float32)

    B, Gc, R = bg[..., 0], bg[..., 1], bg[..., 2]
    Y = luma(bg)
    cream = roi & (Y > 58) & (R > Gc - 2) & (R > B + 8) & (Gc < R + 20)
    chair = (Y < 70) & (R > 40) & (R < 90) & (np.abs(R - Gc) < 20)  # rough
    hole = cream.astype(np.uint8) * (1 - chair.astype(np.uint8))
    hole = cv2.dilate(hole, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    hole = hole * roi.astype(np.uint8)
    # nearest leafy fill (no Telea fog)
    leaf_now = (roi & (Gc > R + 4) & (Gc > B) & (Y < 125)).astype(np.uint8)
    if int(hole.sum()) > 20 and int(leaf_now.sum()) > 20:
        inv = (1 - leaf_now) * 255
        inv = inv.astype(np.uint8)
        # distance to leaf + indexes
        # cv2.distanceTransform with label isn't in all builds; use
        # inpaint only on tiny leftover after copying nearest by blur of leaf
        # Weighted copy: repeatedly dilate leaf colours into holes
        filled = bg.copy()
        mask = hole.astype(bool)
        grow = leaf_now.copy()
        col = bg.copy()
        for _ in range(40):
            if not mask.any():
                break
            dil = cv2.dilate(grow, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
            edge = (dil > 0) & (grow == 0) & mask
            if not edge.any():
                break
            # average of already-grown neighbours
            for c in range(3):
                sm = cv2.blur(col[..., c] * grow.astype(np.float32), (5, 5))
                wt = cv2.blur(grow.astype(np.float32), (5, 5)) + 1e-6
                filled[..., c][edge] = (sm / wt)[edge]
            col = filled
            grow = dil
            mask = mask & (~edge)
        m = cv2.GaussianBlur(hole.astype(np.float32), (0, 0), 1.0)
        m = np.clip(m, 0, 1) * roi.astype(np.float32)
        bg = bg * (1.0 - m[..., None]) + filled * m[..., None]
    return np.clip(bg, 0, 255)


def body_haunch_mask(mattes, tail):
    """Static body (no tail) so haunch fur occludes the root. No ruler cut."""
    a = T.norm_alpha(mattes[0]).astype(np.float32)
    # union a few frames so the body is solid even if matte flickers
    for i in (12, 40, 80, 160):
        a = np.maximum(a, T.norm_alpha(mattes[i]).astype(np.float32))
    rest = tail['rest']
    arc, tot = Hh._arc(rest)
    s, sd, _, _ = Hh._closest(Xg, Yg, rest, arc, tot)
    w = np.interp(np.clip(s, 0, 1),
                  np.linspace(0, 1, len(tail['widths'])), tail['widths'])
    # Distal tail only — keep the root inside the body occluder so the
    # backward hose is hidden under the haunch/belly fur.
    on_tail = (sstep((w + 16.0 - np.abs(sd)) / 4.0)
               * sstep((s - 0.16) / 0.08)
               * sstep((0.98 - s) / 0.08))
    body = np.clip(a * (1.0 - 0.96 * on_tail), 0, 1)
    body = body * sstep((Yg - 500.0) / 14.0) * sstep((Xg - 860.0) / 14.0)
    # furry edge, no straight vertical clip
    rng = np.random.RandomState(7)
    noise = cv2.GaussianBlur(rng.randn(H, W).astype(np.float32), (0, 0), 2.2)
    edge = cv2.GaussianBlur(body, (0, 0), 1.4)
    body = np.clip(edge + 0.10 * noise * (edge * (1.0 - edge) * 4.0), 0, 1)
    return np.clip(body, 0, 1)


def apply_soft_tail(img, cat_a, ph, tail, bg, occ, plate, mic, body):
    """Clear inherited hook, restore leaves, draw a soft-edged painted tail
    that tucks behind the haunch.
    """
    imgf = img.astype(np.float32)
    rest = tail['rest']
    gate = sstep((Yg - 500.0) / 16.0) * sstep((Xg - 900.0) / 16.0)
    occ_s = np.clip(occ * gate, 0, 1)
    # never punch the plate through body fur (dilated hook occ can overlap the hip)
    occ_s = occ_s * (1.0 - np.clip(body, 0, 1))
    torso = (np.clip(cat_a, 0, 1) * sstep((930.0 - Xg) / 16.0)
             * sstep((Yg - 480.0) / 16.0))
    occ_s = occ_s * (1.0 - np.clip(torso, 0, 1))
    cleared = imgf * (1.0 - occ_s[..., None]) + bg * occ_s[..., None]
    # also put the static plate on any leftover plant-corner occupancy
    live = np.clip(plate, 0, 1) * gate
    live = live * (1.0 - sstep((np.clip(cat_a, 0, 1) - 0.25) / 0.25))
    live = cv2.GaussianBlur(live, (0, 0), 1.0)
    rest_mix = live * (1.0 - np.clip(cat_a, 0, 1))
    cleared = cleared * (1.0 - rest_mix[..., None]) + bg * rest_mix[..., None]

    defp = Wx.deform_flicks(rest, ph)
    tip = defp[-1]
    cleared = Wx.apply_plant_reaction(cleared, live, ph, tip)

    pad = 90
    xs = np.concatenate([rest[:, 0], defp[:, 0]])
    ys = np.concatenate([rest[:, 1], defp[:, 1]])
    x0, x1 = int(max(0, xs.min() - pad)), int(min(W, xs.max() + pad + 1))
    y0, y1 = int(max(0, ys.min() - pad)), int(min(H, ys.max() + pad + 1))
    moved = float(np.max(np.abs(defp - rest)))
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    arc_d, tot_d = Hh._arc(defp)
    s, sd, tx, ty = Hh._closest(xx, yy, defp, arc_d, tot_d)
    if moved < 0.4:
        rgb = tail['world_rgb'][y0:y1, x0:x1].copy()
        al = tail['world_a'][y0:y1, x0:x1].copy()
    else:
        arc_r, tot_r = Hh._arc(rest)
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
                  np.linspace(0, 1, len(tail['widths'])), tail['widths'])
    cx = np.interp(s * tot_d, arc_d, defp[:, 0])
    cy = np.interp(s * tot_d, arc_d, defp[:, 1])
    along = (xx - cx) * tx + (yy - cy) * ty
    al = np.clip(al, 0, 1)
    # soft fur fringe (noise along the tangent), not a cookie-cutter
    fluff = 5.5 + 3.0 * np.sin(s * 47.0) + 2.0 * np.sin(s * 113.0 + sd * 0.35)
    soft = 4.8 + 1.8 * (0.5 + 0.5 * np.sin(s * 71.0))
    al = al * sstep((w + fluff - np.abs(sd)) / np.maximum(soft, 1.0))
    tipz = sstep((s - 0.82) / 0.10)
    rcap = np.hypot(np.maximum(along, 0.0), sd)
    al = al * (1.0 - tipz + tipz * sstep(((w + 7.0) - rcap) / 4.0))
    # Kill the backward hose at the root: _closest clamps s to 0, so pixels
    # behind the first vertebra still sample the root cross-section.
    rootz = 1.0 - sstep((s - 0.08) / 0.08)
    al = al * (1.0 - rootz * sstep((-along) / 2.5))
    # drop clearly green leaves baked into the sprite
    leaf = ((rgb[..., 1] > rgb[..., 2] + 14)
            & (rgb[..., 1] > rgb[..., 0] + 8)
            & (rgb[..., 1] > 40))
    al = al * (1.0 - 0.95 * leaf.astype(np.float32))
    # keep a soft edge — do not re-harden with a steep sstep
    al = np.clip(al, 0, 1)
    la = np.zeros((H, W), np.float32)
    layer = np.zeros((H, W, 3), np.float32)
    la[y0:y1, x0:x1] = al
    layer[y0:y1, x0:x1] = rgb
    # wherever the new tail is absent, the cleared plate (leaves) stays
    comp = cleared * (1.0 - la[..., None]) + layer * la[..., None]
    # root behind the haunch: body fur occludes the join
    haunch = np.clip(body, 0, 1)
    comp = comp * (1.0 - haunch[..., None]) + imgf * haunch[..., None]
    comp = imgf * mic[..., None] + comp * (1.0 - mic[..., None])
    area = float((la * (1.0 - haunch) > 0.45).sum())
    return np.clip(comp, 0, 255), tip, area, la


# ---------------- mouth transfer ----------------
def load_ref_bank(ids):
    cap = cv2.VideoCapture(str(REF2))
    need = set(ids)
    out = {}
    i = 0
    while need:
        ok, im = cap.read()
        if not ok:
            break
        if i in need:
            out[i] = im
            need.remove(i)
        i += 1
    cap.release()
    return out


def key_ref(im):
    imf = im.astype(np.float32)
    hsv = cv2.cvtColor(np.clip(im, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV)
    Hh_, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    B, G, R = imf[..., 0], imf[..., 1], imf[..., 2]
    green = ((Hh_ >= 35) & (Hh_ <= 85) & (S > 40) & (V > 40)) | (
        (G > R + 18) & (G > B + 15) & (G > 60)
    )
    Y = luma(imf)
    mic = ((Y < 95) & (~green) & (np.minimum(R, np.minimum(G, B)) < 110)
           & (np.arange(im.shape[0])[:, None] > 210))
    mic = mic.astype(np.uint8)
    mic = cv2.morphologyEx(mic, cv2.MORPH_CLOSE,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    mic = cv2.dilate(mic, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(mic, 8)
    best, by = 0, -1
    for k in range(1, n):
        cy = st[k, cv2.CC_STAT_TOP] + st[k, cv2.CC_STAT_HEIGHT] * 0.5
        if st[k, cv2.CC_STAT_AREA] > 400 and cy > by:
            by, best = cy, k
    mic = ((lab == best) if best else mic).astype(np.uint8)
    fg = (~green).astype(np.float32)
    fg = fg * (1.0 - cv2.GaussianBlur(mic.astype(np.float32), (0, 0), 2.0))
    fg = cv2.erode(fg, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    fg = cv2.GaussianBlur(fg, (0, 0), 1.0)
    return imf, np.clip(fg, 0, 1), mic.astype(np.float32)


def sim0(sm, dm, scale):
    """Scale + translate (similarity with angle 0). Forward matrix for this
    OpenCV build's warpAffine (empirically src->dst)."""
    M = np.zeros((2, 3), np.float32)
    M[0, 0] = M[1, 1] = scale
    M[:, 2] = dm - scale * sm
    return M


def reinhard_lab(src, dst, ms, md):
    s = cv2.cvtColor(np.clip(src, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    d = cv2.cvtColor(np.clip(dst, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    out = s.copy()
    ms = (ms > 0.4)
    md = (md > 0.4)
    if ms.sum() < 30 or md.sum() < 30:
        return src
    for c in range(3):
        mu_s, sd_s = float(s[..., c][ms].mean()), float(s[..., c][ms].std()) + 1e-6
        mu_d, sd_d = float(d[..., c][md].mean()), float(d[..., c][md].std()) + 1e-6
        out[..., c] = (s[..., c] - mu_s) * (0.55 * sd_d / sd_s + 0.45) + mu_d
    out = np.clip(out, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_LAB2BGR).astype(np.float32)


def painterly(im, dest, mask):
    u = np.clip(im, 0, 255).astype(np.uint8)
    sm = cv2.bilateralFilter(u, 7, 28, 5).astype(np.float32)
    gdest = dest.astype(np.float32)
    grain = gdest - cv2.GaussianBlur(gdest, (0, 0), 1.15)
    out = sm + 0.70 * grain
    a = mask[..., None]
    return im * (1.0 - a) + np.clip(out, 0, 255) * a


def close_old_mouth(img, slit):
    """Inpaint the inherited sticker slit, tight to the cavity."""
    imgf = img.astype(np.float32)
    y0, y1, x0, x1 = 375, 455, 620, 755
    sub = imgf[y0:y1, x0:x1]
    Y = luma(sub)
    B, G, R = sub[..., 0], sub[..., 1], sub[..., 2]
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    dark = (Y < 80) & (Y > 8)
    pink = (R > G + 16) & (Y < 130) & (Y > 20)
    ell = ((((xx - slit[0]) / 36.0) ** 2) + (((yy - slit[1]) / 20.0) ** 2) < 1.0)
    op = ((dark | pink) & ell).astype(np.uint8)
    op = cv2.dilate(op, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    if int(op.sum()) < 8:
        return img
    fill = cv2.inpaint(np.clip(sub, 0, 255).astype(np.uint8),
                       op * 255, 6, cv2.INPAINT_TELEA).astype(np.float32)
    m = np.clip(cv2.GaussianBlur(op.astype(np.float32), (0, 0), 1.1), 0, 1)
    out = imgf.copy()
    out[y0:y1, x0:x1] = sub * (1.0 - m[..., None]) + fill * m[..., None]
    return np.clip(out, 0, 255)


def chin_drop(img, dy, nose, eyes, ears):
    """Lower lip / chin / upper throat move down as one piece. Tight sides."""
    if dy < 0.3:
        return img
    imgf = img.astype(np.float32)
    r2 = ((Xg - float(nose[0])) / 30.0) ** 2 + ((Yg - 490.0) / 50.0) ** 2
    m = sstep((1.0 - r2) / 0.22)
    m = m * sstep((Yg - 438.0) / 8.0) * (1.0 - sstep((Yg - 555.0) / 10.0))
    m = m * (1.0 - eyes) * (1.0 - ears)
    mapy = np.clip(Yg - float(dy) * m, 0, H - 1).astype(np.float32)
    moved = cv2.remap(imgf, Xg, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    out = imgf * (1.0 - m[..., None]) + moved * m[..., None]
    lock = np.clip(np.maximum(eyes, ears), 0, 1)
    lock = np.maximum(lock, 1.0 - sstep((Yg - 398.0) / 5.0))
    return np.clip(out * (1.0 - lock[..., None]) + imgf * lock[..., None], 0, 255)


def transfer_oral(dst, ref_im, rn, rm, dn, dm, amt, tight=False):
    src, fg, mic = key_ref(ref_im)
    dist_s = float(np.linalg.norm(rm - rn)) + 1e-8
    dist_d = float(np.linalg.norm(dm - dn))
    scale = float(np.clip(dist_d / dist_s, 0.55, 0.85))
    M = sim0(rm, dm, scale)
    h, w = src.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ax, ay = (44.0, 36.0) if tight else (52.0, 44.0)
    r2 = ((xx - rm[0]) / ax) ** 2 + ((yy - rm[1]) / ay) ** 2
    muzzle = sstep((1.0 - r2) / 0.22) * sstep((yy - (rn[1] + 14.0)) / 5.0) * fg
    muzzle = muzzle * (1.0 - cv2.GaussianBlur(mic, (0, 0), 1.5))
    muzzle = muzzle.astype(np.float32)
    rgb = cv2.warpAffine(src, M, (W, H), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT)
    a = cv2.warpAffine(muzzle, M, (W, H), flags=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT)
    a = np.clip(a, 0, 1)
    Ys = luma(src)
    cav = (Ys < 80) & (muzzle > 0.3)
    fur_s = cv2.warpAffine(((muzzle > 0.4) & (~cav) & (Ys > 80)).astype(np.float32),
                           M, (W, H), flags=cv2.INTER_LINEAR)
    Yd = luma(dst)
    fur_d = ((((Xg - dn[0]) / 48.0) ** 2 + ((Yg - (dn[1] + 28.0)) / 36.0) ** 2) < 1.0)
    fur_d = fur_d & (Yd > 70) & (Yd < 190) & (Yg > dn[1] + 8)
    fur_d = fur_d.astype(np.float32)
    rgb = reinhard_lab(rgb, dst, fur_s, fur_d)
    rgb = painterly(rgb, dst, a)
    if (fur_s > 0.4).sum() > 10 and (fur_d > 0.4).sum() > 10:
        ys = float(luma(rgb)[fur_s > 0.4].mean())
        yd = float(Yd[fur_d > 0.4].mean())
        if ys > 1:
            rgb = np.clip(rgb * (0.40 + 0.60 * (yd / ys)), 0, 255)
    a = a * float(amt)
    binm = (a > 0.12).astype(np.uint8)
    a = a * sstep(cv2.distanceTransform(binm, cv2.DIST_L2, 3) / 7.0)
    a = a * sstep((Yg - (dn[1] + 16.0)) / 5.0)
    a = a * sstep((dm[0] + 52.0 - Xg) / 6.0) * sstep((Xg - (dm[0] - 52.0)) / 6.0)
    a = a * sstep(((dm[1] + 36.0) - Yg) / 6.0)
    a = np.clip(a, 0, 1)
    out = dst.astype(np.float32) * (1.0 - a[..., None]) + rgb * a[..., None]
    return np.clip(out, 0, 255)


def apply_meow(img, i, eyes, ears, refs):
    a, b = Wx.meow_amount(i)
    amt = max(a, b)
    if amt < 0.04:
        return img
    locked = img.copy()
    # medium uses the smaller 0:14-ish n245 oval; big uses n250
    # n245 landed as a pale egg beside the slit; n250 is the donor that
    # actually registers. Medium uses a tighter crop of the same opening.
    if b >= a:
        ri, dm, dy, tight = 250, U_MOUTH_B, 6.0 * b, False
    else:
        ri, dm, dy, tight = 250, U_MOUTH_A, 4.5 * a, True
    # ease dest mouth slightly down as the jaw opens
    dm = dm.copy()
    dm[1] = dm[1] + 2.0 * amt
    rn, rm = REF_LM[ri]
    work = close_old_mouth(img, dm)
    work = chin_drop(work, dy, U_NOSE, eyes, ears)
    work = transfer_oral(work, refs[ri], rn, rm, U_NOSE, dm, amt, tight=tight)
    lock = np.clip(np.maximum(eyes, ears), 0, 1)
    work = work * (1.0 - lock[..., None]) + locked.astype(np.float32) * lock[..., None]
    return np.clip(work, 0, 255)


def process_one(src_u, cat_a, ph, ctx, i):
    img, tip, area, la = apply_soft_tail(
        src_u, cat_a, ph, ctx['tail'], ctx['tail_bg'], ctx['occ'],
        ctx['plate'], ctx['mic'], ctx['body'])
    img = apply_meow(img, i, ctx['eyes'], ctx['ears'], ctx['refs'])
    y0, y1, x0, x1 = TITLE
    img = img.copy()
    img[y0:y1, x0:x1] = src_u[y0:y1, x0:x1]
    return np.clip(img, 0, 255).astype(np.uint8), tip, area


def process(src_u, mattes, ctx, preview=None):
    want = set(preview) if preview else set(range(NFR))
    out = [None] * NFR
    areas = np.full(NFR, np.nan, np.float32)
    tips = [None] * NFR
    for i in range(NFR):
        if i not in want:
            continue
        ph = Hh.phase_of(i)
        out[i], tips[i], areas[i] = process_one(
            src_u[i], T.norm_alpha(mattes[i]), ph, ctx, i)
        if preview or i % 25 == 0:
            print(f'  n{i} ph={ph} tip={tips[i]} area={areas[i]:.0f}', flush=True)
    return out, areas, tips


# ---------------- QC (labels use VER) ----------------
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
    try:
        T.face_compare_grid(out, src_u, dest)
    except Exception as e:
        print('face grid skip', e)
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


def encoded_stills(mp4, dest):
    dest.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(mp4))
    frames = {}
    i = 0
    want = {0, 12, 120, 138, 180, 432}
    while True:
        ok, im = cap.read()
        if not ok:
            break
        if i in want:
            frames[i] = im
        i += 1
    cap.release()
    for i, box, name, sc in (
        (0, (380, 720, 880, 1280), 'encoded-tail-n0', 2),
        (12, (380, 720, 880, 1280), 'encoded-tail-n12', 2),
        (120, (380, 720, 880, 1280), 'encoded-tail-n120', 2),
        (180, (380, 720, 880, 1280), 'encoded-tail-n180', 2),
        (138, (330, 500, 620, 860), 'encoded-mouth-n138', 3),
        (432, (330, 500, 620, 860), 'encoded-mouth-n432', 3),
        (138, (200, 360, 580, 900), 'encoded-face-n138', 2),
        (432, (200, 360, 580, 900), 'encoded-face-n432', 2),
    ):
        if i not in frames:
            continue
        y0, y1, x0, x1 = box
        z = cv2.resize(frames[i][y0:y1, x0:x1], None, fx=sc, fy=sc,
                       interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(dest / f'{name}.png'), z)
    # also dump raw encoded frames used above
    for i in want:
        if i in frames:
            cv2.imwrite(str(dest / f'encoded-n{i}.png'), frames[i])
    return frames


def write_qc(out, src_u, dest, areas, tips, rest_tip, mp4=None):
    dest.mkdir(parents=True, exist_ok=True)
    tail_strip_2x(out, dest, every=6)
    meow_seq(out, dest)
    eye_seq(out, dest)
    before_after(out, src_u, dest)
    face_and_chest(out, src_u, dest)
    area_report(areas, dest, tips, rest_tip)
    try:
        T.mouth_qc(out, src_u, dest)
    except Exception as e:
        print('mouth_qc skip', e)
    if mp4 is not None:
        encoded_stills(mp4, dest)


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
        ('v41w-ex-mp4', P / 'v41w-ex-test-25s.mp4', V41W_SHA),
        ('pcm', PCM, Hh.PCM_SHA),
    ):
        got = sha256(path)
        if got != expect:
            raise SystemExit(f'{label} sha changed: {got}')
    print('frozen shas ok', flush=True)

    print('loading u-ex / q-ex / mattes / refs…', flush=True)
    src_u = Hh.load_span(SRC_U)
    src_q = Hh.load_span(SRC_Q)
    mattes = Hh.load_span(MATTE, gray=True)
    assert len(src_u) == NFR
    refs = load_ref_bank([245, 250])
    assert 245 in refs and 250 in refs

    print('preparing tail plates…', flush=True)
    tail = T.prep_tail_padded(T.SPRITE_FIXED)
    rest = T.resample_poly(tail['rest'], 20)
    tail['rest'] = rest
    mic = Hh._mic_mask()
    tail_bg, occ = T.build_tail_bg(src_q, mattes, tail)
    plate = G.build_plant_plate(src_q[121], T.norm_alpha(mattes[121]))
    leaf_plate = build_full_leaf_plate(src_q, mattes, tail_bg)
    corner = sstep((Xg - 980.0) / 10.0) * sstep((Yg - 510.0) / 10.0)
    tail_bg = tail_bg * (1.0 - corner[..., None]) + leaf_plate * corner[..., None]
    hook_occ = Wx.inherited_hook_occ(src_u, mattes)
    occ = np.maximum(occ, hook_occ)
    occ = np.clip(cv2.dilate((occ > 0.08).astype(np.uint8),
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))).astype(np.float32), 0, 1)
    occ = cv2.GaussianBlur(occ, (0, 0), 3.0)
    body = body_haunch_mask(mattes, tail)
    eyes = Wx.eye_mask()
    ears = Wx.ear_mask()
    ctx = dict(tail=tail, tail_bg=tail_bg, occ=occ, plate=plate, mic=mic,
               body=body, eyes=eyes, ears=ears, refs=refs)
    QC.mkdir(parents=True, exist_ok=True)

    if args.preview:
        idx = sorted(set(list(range(0, 241)) + list(range(420, 450))))
        out, areas, tips = process(src_u, mattes, ctx, preview=idx)
        dest = QC / 'preview'
        write_qc(out, src_u, dest, areas, tips, rest[-1], mp4=None)
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
    shutil.copy2(Path(__file__), WORK / 'build_v41x_ex.py')
    print('v41u-ex still', sha256(P / 'v41u-ex-test-25s.mp4'), flush=True)
    dest = QC / 'encoded-tmp'
    write_qc(out, src_u, dest, areas, tips, rest[-1], mp4=mp4)


if __name__ == '__main__':
    main()
