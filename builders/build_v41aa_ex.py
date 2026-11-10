#!/usr/bin/env python3
"""v41aa-ex: organic secondary motion on frozen v41z-ex.

Keeps z-ex meows (tongue and fangs, chin-hinge), tail flicks, seam morph,
plates, snow, steam, lamp, title and audio. Adds gentle eased mesh warps
on the cat layer only: breathing, lagged head drift, meow anticipation,
and a couple of single-ear twitches.

The mouth reference is a swap-in hook (apply_mouth / MOUTH_REF) and is a
no-op until a path is provided.

Does not edit earlier builders or encodes.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

P = Path('/workspace/lofi-v41-meow')
sys.path.insert(0, str(P))
import build_v41h as Hh
import build_v41y_ex as Y
import build_v41z_ex as Z

VER = 'v41aa-ex'
WORK = P / VER
QC = P / 'qc' / VER
SRC_U = Y.SRC_U
SRC_Z = P / 'v41z-ex' / 'v41z-ex-span-890-915-lossless.mkv'
PCM = Hh.PCM

V41U_SHA, V41U_LOS_SHA = Z.V41U_SHA, Z.V41U_LOS_SHA
V41Z_SHA = 'a1ae8a43bbd47cab687bbeb0f21613163c16b400a160c6836e2d0a6ef11d34c5'
V41Z_LOS_SHA = 'e769a0f6d2871c96fa057b8bf3156ee6c7219063801a85d30b63c07834e16a34'

NFR, W, H, NCYCLE = 600, 1280, 720, 241
TITLE = Hh.TITLE
HEAD = Z.HEAD
Xg, Yg = Y.Xg, Y.Yg

# 3 breaths in the 241-frame loop ≈ 3.35 s each (1 per 3–4 s), seams at k≈0.
N_BREATH = 3
CHEST_AMP = 1.80          # px vertical at full inhale
HEAD_BREATH_AMP = 0.55    # lagged, smaller
HEAD_LAG = 8              # frames
DRIFT_AMP_X, DRIFT_AMP_Y = 0.55, 0.40

# anticipation (px)
ANT_PRE, ANT_POST = 10, 14
ANT_LIFT = 1.60
ANT_EAR = 2.40

# ear bases (n0), and twitch schedule (absolute frames / cycle-local)
L_BASE = (622.0, 246.0)
R_BASE = (806.0, 248.0)
TWITCHES = [
    # (start, duration, degrees, 'L'|'R', predicate)
    # gen1 rest, left ear; also fires on the third (gen1) cycle at n553
    {'start_local': 72, 'dur': 5, 'deg': 2.5, 'side': 'L', 'gen2': False},
    # gen2 rest, right ear
    {'start_local': 108, 'dur': 6, 'deg': 2.6, 'side': 'R', 'gen2': True},
]

# Optional mouth reference. Env or CLI --mouth-ref. Swap-in, no-op if unset.
MOUTH_REF = os.environ.get('V41AA_MOUTH_REF') or None

sha256 = Z.sha256
luma = Z.luma
mad_rgb, mad_luma = Z.mad_rgb, Z.mad_luma


def smootherstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def breath_k(local):
    """Asymmetric inhale (40%) / exhale (60%), 3 cycles per 241, k(0)≈k(240)≈0."""
    p = (float(local) / float(NCYCLE)) * N_BREATH
    frac = p - np.floor(p)
    if frac < 0.40:
        return float(smootherstep(frac / 0.40))
    return float(1.0 - smootherstep((frac - 0.40) / 0.60))


# precompute 241-periodic signals
K_BREATH = np.array([breath_k(i) for i in range(NCYCLE)], np.float32)
K_HEAD = np.roll(K_BREATH, HEAD_LAG)  # lag the body, still periodic


def drift(local):
    """Slow 241-periodic wander; ~0 at the loop join."""
    th = 2.0 * np.pi * local / NCYCLE
    dx = DRIFT_AMP_X * (0.70 * np.sin(th) + 0.30 * np.sin(2.0 * th + 1.1))
    dy = DRIFT_AMP_Y * (0.65 * np.cos(th + 0.35) + 0.25 * np.sin(3.0 * th + 2.0)
                        - 0.65 * np.cos(0.35))  # subtract DC so dy(0)=0
    return float(dx), float(dy)


def pulse(t01):
    """Eased bump, 0 at ends, 1 at mid (C1, not linear)."""
    t01 = np.clip(t01, 0.0, 1.0)
    return float(np.sin(np.pi * t01) ** 2)


def anticip(local, peak):
    """Head lift + ear-forward around a meow peak. 0 outside the window."""
    lo, hi = peak - ANT_PRE, peak + ANT_POST
    if local < lo or local > hi:
        return 0.0, 0.0
    if local <= peak:
        w = smootherstep((local - lo) / max(ANT_PRE, 1))
    else:
        w = 1.0 - smootherstep((local - peak) / max(ANT_POST, 1))
    return -ANT_LIFT * w, ANT_EAR * w


def twitch_angle(i, local, use2):
    """Signed radians for L and R ears at output frame i."""
    ang_L = 0.0
    ang_R = 0.0
    for tw in TWITCHES:
        if tw['gen2'] != use2:
            continue
        s, d = tw['start_local'], tw['dur']
        if s <= local < s + d:
            w = pulse((local - s + 0.5) / d)
            a = np.deg2rad(tw['deg']) * w
            if tw['side'] == 'L':
                ang_L += a
            else:
                ang_R += a
    return ang_L, ang_R


def build_weight_maps():
    """Soft regional weights. Head is flat across the face so eyes follow the skull."""
    def blob(cx, cy, sx, sy):
        return np.exp(-0.5 * ((Xg - cx) / sx) ** 2 - 0.5 * ((Yg - cy) / sy) ** 2)

    face = np.zeros((H, W), np.float32)
    face[190:410, 540:900] = 1.0
    face = cv2.GaussianBlur(face, (0, 0), 18.0)
    head = np.clip(np.maximum(blob(710, 300, 170, 140), 0.92 * face), 0, 1)
    # fade head out below the neck so the chest can move independently
    neck = smootherstep((440.0 - Yg) / 50.0)
    head = head * neck

    chest = blob(700, 515, 210, 150)
    chest = chest * smootherstep((Yg - 370.0) / 50.0)
    # don't let chest weights grab the tail/plant corner
    chest = chest * smootherstep((1020.0 - Xg) / 40.0)

    ear_L = blob(598, 200, 58, 52)
    ear_R = blob(848, 205, 60, 54)
    # keep ear weights from spilling into the window
    ear_L = ear_L * smootherstep((Xg - 530.0) / 16.0)
    ear_R = ear_R * smootherstep((920.0 - Xg) / 16.0)

    # flatten head over the eye boxes (lock eyes to skull: uniform translation)
    eyes = np.zeros((H, W), np.float32)
    eyes[240:325, 600:860] = 1.0
    eyes = cv2.GaussianBlur(eyes, (0, 0), 10.0)
    # raise head weight to almost 1 there, then re-normalize later
    head = np.clip(np.maximum(head, 0.97 * eyes * head.max()), 0, 1)

    return {'head': head, 'chest': chest, 'ear_L': ear_L, 'ear_R': ear_R, 'eyes': eyes}


WEIGHTS = None


def weights():
    global WEIGHTS
    if WEIGHTS is None:
        WEIGHTS = build_weight_maps()
    return WEIGHTS


def rotation_disp(bx, by, ang):
    if abs(ang) < 1e-8:
        return np.zeros((H, W), np.float32), np.zeros((H, W), np.float32)
    ca, sa = np.cos(ang), np.sin(ang)
    rx, ry = Xg - bx, Yg - by
    dx = (ca - 1.0) * rx - sa * ry
    dy = sa * rx + (ca - 1.0) * ry
    return dx.astype(np.float32), dy.astype(np.float32)


def warp_field(i):
    """Displacement (dx, dy) of cat pixels at output frame i, in pixels."""
    use2, local = Y.cycle_of(i)
    wt = weights()
    k_c = float(K_BREATH[local])
    k_h = float(K_HEAD[local])
    drift_x, drift_y = drift(local)
    # meow peak is local 138 on gen1 cycles, local 191 on gen2
    peak = 191 if use2 else 138
    ant_dy, ant_ear = anticip(local, peak)
    ang_L, ang_R = twitch_angle(i, local, use2)

    dx = np.zeros((H, W), np.float32)
    dy = np.zeros((H, W), np.float32)

    # chest / shoulders: rise on inhale, slight expand
    dy = dy + wt['chest'] * (-CHEST_AMP * k_c)
    dx = dx + wt['chest'] * (0.35 * k_c * np.clip((Xg - 700.0) / 200.0, -1, 1))

    # head: lagged breath + wander + anticipation (uniform over the face)
    dy = dy + wt['head'] * (-HEAD_BREATH_AMP * k_h + drift_y + ant_dy)
    dx = dx + wt['head'] * (drift_x)

    # ears forward (anticipation): opposite rotations about each base
    if abs(ant_ear) > 1e-6:
        # 2.4 px tip travel ≈ 2.0 deg at a ~70 px ear length
        a = np.deg2rad(2.0) * (ant_ear / max(ANT_EAR, 1e-6))
        dxl, dyl = rotation_disp(*L_BASE, +a)
        dxr, dyr = rotation_disp(*R_BASE, -a)
        dx = dx + wt['ear_L'] * dxl + wt['ear_R'] * dxr
        dy = dy + wt['ear_L'] * dyl + wt['ear_R'] * dyr

    # single-ear twitches
    if abs(ang_L) > 1e-8:
        dxl, dyl = rotation_disp(*L_BASE, ang_L)
        dx = dx + wt['ear_L'] * dxl
        dy = dy + wt['ear_L'] * dyl
    if abs(ang_R) > 1e-8:
        dxr, dyr = rotation_disp(*R_BASE, ang_R)
        dx = dx + wt['ear_R'] * dxr
        dy = dy + wt['ear_R'] * dyr

    # Plant the paws: sitting cats breathe in the chest, they do not slide
    # on the desk. Fade the field out below the wrists.
    planted = smootherstep((620.0 - Yg) / 50.0).astype(np.float32)
    dx, dy = dx * planted, dy * planted

    return dx, dy


def apply_mouth(out, i, *, z_frame, alpha, src_u):
    """Swap-in mouth step. No-op until MOUTH_REF points at a usable file."""
    ref = MOUTH_REF
    if not ref:
        return out
    path = Path(ref)
    if not path.exists():
        print(f'warning: mouth ref missing {path}', flush=True)
        return out
    return _apply_mouth_reference(out, i, path, z_frame, alpha, src_u)


def _apply_mouth_reference(out, i, path, z_frame, alpha, src_u):
    """Implement when the user path arrives. Identity for now so the hook is safe."""
    return out


def replay_z_alpha(i, g1, g2, a1, a2, start1, start2):
    use2, local = Y.cycle_of(i)
    _gen, al = Z.sample_gen(use2, local, g1, g2, a1, a2)
    w = Z.in_seam(i)
    if w is not None:
        tgt_g, tgt_a = start2 if Z.next_start_is_gen2(i) else start1
        _gen, al = Z.flow_blend(_gen, tgt_g, al, tgt_a, w)
    return np.clip(al.astype(np.float32), 0, 1)


def load_or_build_alpha(g1, g2, a1, a2, cache):
    if cache.exists():
        print('loading cached 600-frame z alpha', flush=True)
        return np.load(cache)['alpha'].astype(np.float32) / 255.0
    print('replaying z-ex alpha for 600 frames…', flush=True)
    start1 = (g1[0].astype(np.float32), a1[0].astype(np.float32))
    start2 = (g2[0].astype(np.float32), a2[0].astype(np.float32))
    out = np.zeros((NFR, H, W), np.float32)
    for i in range(NFR):
        out[i] = replay_z_alpha(i, g1, g2, a1, a2, start1, start2)
        if i % 100 == 0:
            print(f'  alpha {i}', flush=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, alpha=(out * 255).astype(np.uint8))
    return out


def plant_only(alpha_i, z_i):
    """Soft mask of right-corner plant that is not cream tail/fur."""
    roi = (Xg > 970) & (Yg > 460)
    Yl = luma(z_i)
    B, G, R = z_i[..., 0].astype(np.float32), z_i[..., 1].astype(np.float32), z_i[..., 2].astype(np.float32)
    leaf = roi & (G > R + 2) & (G > B - 8) & (Yl < 140)
    m = (leaf.astype(np.float32) * (alpha_i > 0.2))
    m = cv2.GaussianBlur(m, (0, 0), 2.0)
    return np.clip(m, 0, 1)


def compose_warped(z, u, alpha, dx, dy):
    """Warp the z-ex cat onto u-ex-revealed background; plant stays put."""
    a = np.clip(alpha.astype(np.float32), 0, 1)
    p = plant_only(a, z)
    a_cat = np.clip(a - p, 0, 1)
    # do not warp plant or background (zero field there already via weights;
    # still punch the remap so leaves don't smear)
    mx = (Xg - dx).astype(np.float32)
    my = (Yg - dy).astype(np.float32)
    z_w = cv2.remap(z, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    a_w = cv2.remap(a_cat, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    a_w = np.clip(a_w, 0, 1)
    # uncover original cat with u-ex (snow/desk/window stay u, no ghost fur)
    dil = cv2.dilate((a_cat > 0.08).astype(np.uint8),
                     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    dil = cv2.GaussianBlur(dil.astype(np.float32), (0, 0), 2.0)
    bg = z.astype(np.float32) * (1.0 - dil[..., None]) + u.astype(np.float32) * dil[..., None]
    out = bg * (1.0 - a_w[..., None]) + z_w.astype(np.float32) * a_w[..., None]
    # unwarped plant back on top (leaves stay)
    out = out * (1.0 - p[..., None]) + z.astype(np.float32) * p[..., None]
    # exact u/z copy outside the matte
    off = (a_w < 0.02) & (p < 0.02) & (dil < 0.02)
    out[off] = z[off].astype(np.float32)
    y0, y1, x0, x1 = TITLE
    out[y0:y1, x0:x1] = z[y0:y1, x0:x1]
    return np.clip(out, 0, 255).astype(np.uint8)


def process(src_z, src_u, alpha, preview=None):
    want = set(preview) if preview else set(range(NFR))
    out = [None] * NFR
    for i in range(NFR):
        if i not in want:
            continue
        dx, dy = warp_field(i)
        # kill any residual field outside a generous cat prior
        a = alpha[i]
        gate = cv2.dilate((a > 0.05).astype(np.uint8),
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
        gate = cv2.GaussianBlur(gate.astype(np.float32), (0, 0), 4.0)
        dx, dy = dx * gate, dy * gate
        im = compose_warped(src_z[i], src_u[i], a, dx, dy)
        im = apply_mouth(im, i, z_frame=src_z[i], alpha=a, src_u=src_u[i])
        out[i] = im
        if preview or i % 25 == 0:
            use2, local = Y.cycle_of(i)
            print(f'  n{i} local={local} k={K_BREATH[local]:.2f} '
                  f'dy_chest={-CHEST_AMP*K_BREATH[local]:.2f}', flush=True)
    return out


# ---------------- QC ----------------
def flow_mags(frames, src_u, idxs=None):
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    regs = {
        'head': (520, 900, 160, 430),
        'ears': (520, 900, 150, 250),
        'chest': (520, 920, 400, 620),
        'tail': (880, 1280, 420, 720),
    }
    n = len(frames)
    mags = {k: [] for k in regs}
    last = None
    last_i = None
    for i in range(n - 1):
        if frames[i] is None or frames[i + 1] is None:
            for k in regs:
                mags[k].append(np.nan)
            continue
        g0 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        g1 = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY)
        flow = dis.calc(g0, g1, None)
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        for k, (x0, x1, y0, y1) in regs.items():
            sl = mag[y0:y1, x0:x1]
            d = np.abs(frames[i][y0:y1, x0:x1].astype(np.float32)
                       - src_u[i][y0:y1, x0:x1].astype(np.float32)).mean(axis=2)
            m = d > 2.0
            mags[k].append(float(sl[m].mean()) if m.sum() > 200 else float(sl.mean()))
    return {k: np.array(v, np.float32) for k, v in mags.items()}


def draw_flow_plot(mags_z, mags_aa, dest):
    dest.mkdir(parents=True, exist_ok=True)
    keys = ['head', 'ears', 'chest', 'tail']
    Wd, Hd = 1400, 1000
    plot = np.ones((Hd, Wd, 3), np.uint8) * 245
    l, r, t, b = 70, 20, 36, 40
    pw, ph = Wd - l - r, Hd - t - b
    row_h = ph // 4
    marks = {138: (40, 40, 40), 241: (90, 90, 90), 432: (40, 40, 40), 482: (90, 90, 90)}
    for ri, k in enumerate(keys):
        y0 = t + ri * row_h
        vz, va = mags_z[k], mags_aa[k]
        vmax = max(2.0, float(np.nanpercentile(np.concatenate([vz, va]), 99)) * 1.15)
        cv2.rectangle(plot, (l, y0), (Wd - r, y0 + row_h - 8), (255, 255, 255), -1)
        cv2.rectangle(plot, (l, y0), (Wd - r, y0 + row_h - 8), (200, 200, 200), 1)
        for n, col in marks.items():
            x = l + int(n / 599 * (pw - 1))
            cv2.line(plot, (x, y0), (x, y0 + row_h - 8), col, 1)
        def pts(v, col):
            prev = None
            for i, val in enumerate(v):
                if not np.isfinite(val):
                    prev = None
                    continue
                x = l + int(i / 599 * (pw - 1))
                y = y0 + row_h - 8 - int(np.clip(val / vmax, 0, 1) * (row_h - 16))
                if prev is not None:
                    cv2.line(plot, prev, (x, y), col, 1, cv2.LINE_AA)
                prev = (x, y)
        pts(vz, (80, 80, 200))   # z: reddish in BGR? wait BGR: red = (0,0,200) no (38,92,196) was orange
        pts(va, (80, 140, 40))    # aa green
        cv2.putText(plot, f'{k}  (z orange, aa green)  max {vmax:.2f}',
                    (l + 6, y0 + 16), cv2.FONT_HERSHEY_PLAIN, 1.1, (20, 20, 20), 1)
    cv2.putText(plot, 'flow mag px/frame  marks n138 n241 n432 n482',
                (l, 22), cv2.FONT_HERSHEY_PLAIN, 1.2, (20, 20, 20), 1)
    path = dest / 'flow-mag-z-vs-aa.png'
    cv2.imwrite(str(path), plot)
    print('flow plot', path)
    return path


def edge_qc(out, src_z, src_u, dest):
    dest.mkdir(parents=True, exist_ok=True)
    # confirm window frame / desk outside the cat stay byte-close to z (and u)
    rows = []
    for i in range(0, 600, 10):
        if out[i] is None:
            continue
        d = cv2.absdiff(out[i], src_z[i]).astype(np.float32)
        y0, y1, x0, x1 = TITLE
        title = float(d[y0:y1, x0:x1].mean())
        desk = float(d[400:720, 80:500].mean())
        # a strip of left window frame, away from the ear
        frame = float(d[40:160, 250:360].mean())
        rows.append((i, title, desk, frame, float(d.mean())))
    path = dest / 'edge-mae.txt'
    with open(path, 'w') as f:
        f.write('i title desk left_frame global\n')
        for r in rows:
            f.write(f'{r[0]:4d}  {r[1]:.4f}  {r[2]:.4f}  {r[3]:.4f}  {r[4]:.4f}\n')
    print('edge mae', path,
          'title', np.mean([r[1] for r in rows]),
          'desk', np.mean([r[2] for r in rows]),
          'frame', np.mean([r[3] for r in rows]))
    # visual pairs at rest, meow, twitch, breath peak
    for i, tag in ((0, 'n0'), (32, 'breath-peak'), (72, 'twitch-L'),
                   (138, 'meow138'), (241, 'n241'), (349, 'twitch-R'),
                   (432, 'meow432'), (482, 'n482')):
        if i >= len(out) or out[i] is None:
            continue
        def trip(box, name, sc=2):
            y0, y1, x0, x1 = box
            p = np.hstack([src_z[i][y0:y1, x0:x1], out[i][y0:y1, x0:x1]])
            p = cv2.resize(p, None, fx=sc, fy=sc, interpolation=cv2.INTER_NEAREST)
            cv2.putText(p, f'z {name} {tag}', (8, 24), cv2.FONT_HERSHEY_PLAIN, 1.6, (255, 255, 255), 2)
            cv2.imwrite(str(dest / f'{name}-{tag}.png'), p)
        trip((80, 360, 480, 760), 'window-edge')
        trip((500, 720, 480, 900), 'desk-edge')
        trip((180, 430, 520, 900), 'face')
        trip((140, 280, 520, 920), 'ears')
        trip((400, 640, 520, 920), 'chest')


def breath_strip(out, src_z, dest):
    dest.mkdir(parents=True, exist_ok=True)
    # one breath around local 32 (and 32+80=113) on cycle 0, 2x chest crop
    cells = []
    for i in list(range(16, 50, 2)) + list(range(96, 130, 2)):
        if out[i] is None:
            continue
        a = src_z[i][400:620, 540:900]
        b = out[i][400:620, 540:900]
        pair = np.vstack([a, b])
        cv2.putText(pair, f'z n{i}', (6, 20), cv2.FONT_HERSHEY_PLAIN, 1.3, (255, 255, 255), 1)
        cv2.putText(pair, 'aa', (6, a.shape[0] + 20), cv2.FONT_HERSHEY_PLAIN, 1.3, (255, 255, 255), 1)
        cells.append(pair)
    if cells:
        path = dest / 'breath-chest-strip.png'
        cv2.imwrite(str(path), np.hstack(cells))
        print('breath strip', path)


def twitch_qc(out, src_z, dest):
    dest.mkdir(parents=True, exist_ok=True)
    for start, tag, box in ((70, 'L', (140, 280, 520, 760)),
                            (347, 'R', (140, 280, 700, 960))):
        cells = []
        for i in range(start, start + 10):
            if i >= len(out) or out[i] is None:
                continue
            a = src_z[i][box[0]:box[1], box[2]:box[3]]
            b = out[i][box[0]:box[1], box[2]:box[3]]
            pair = np.vstack([
                cv2.resize(a, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST),
                cv2.resize(b, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST),
            ])
            cv2.putText(pair, f'z n{i}', (6, 22), cv2.FONT_HERSHEY_PLAIN, 1.4, (255, 255, 255), 1)
            cells.append(pair)
        if cells:
            path = dest / f'twitch-{tag}-strip.png'
            cv2.imwrite(str(path), np.hstack(cells))
            print('twitch strip', path)


def seam_report(out, src_z, dest):
    dest.mkdir(parents=True, exist_ok=True)
    typ = list(range(10, 30)) + list(range(50, 70))
    lines = ['pair  z_rgb  z_luma  aa_rgb  aa_luma']
    for a, b in ((239, 240), (240, 241), (241, 242),
                 (480, 481), (481, 482), (482, 483), (0, 1), (10, 11)):
        if out[a] is None or out[b] is None:
            continue
        zr, zl = mad_rgb(src_z[a], src_z[b], HEAD), mad_luma(src_z[a], src_z[b], HEAD)
        ar, al_ = mad_rgb(out[a], out[b], HEAD), mad_luma(out[a], out[b], HEAD)
        lines.append(f'{a}-{b}  {zr:.4f}  {zl:.4f}  {ar:.4f}  {al_:.4f}')
    def typ_of(seq):
        vals = [mad_rgb(seq[i], seq[i + 1], HEAD) for i in typ
                if seq[i] is not None and seq[i + 1] is not None]
        return float(np.mean(vals)) if vals else float('nan')
    lines.append(f'typical(10-30,50-70) z {typ_of(src_z):.4f}  aa {typ_of(out):.4f}')
    path = dest / 'seam-mad.txt'
    path.write_text('\n'.join(lines) + '\n')
    print('SEAM MAD\n' + '\n'.join(lines), flush=True)
    # visual
    def band(im):
        return cv2.resize(im[140:280, 560:940], None, fx=2, fy=2,
                          interpolation=cv2.INTER_NEAREST)
    for a, b, name in ((240, 241, '240-241'), (481, 482, '481-482')):
        if out[a] is None or out[b] is None:
            continue
        row = np.hstack([band(src_z[a]), band(src_z[b]), band(out[a]), band(out[b])])
        cv2.putText(row, f'z{a}', (8, 24), cv2.FONT_HERSHEY_PLAIN, 1.6, (255, 255, 255), 2)
        cv2.putText(row, f'z{b}', (band(src_z[a]).shape[1] + 8, 24),
                    cv2.FONT_HERSHEY_PLAIN, 1.6, (255, 255, 255), 2)
        cv2.putText(row, f'aa{a}', (2 * band(src_z[a]).shape[1] + 8, 24),
                    cv2.FONT_HERSHEY_PLAIN, 1.6, (255, 255, 255), 2)
        cv2.putText(row, f'aa{b}', (3 * band(src_z[a]).shape[1] + 8, 24),
                    cv2.FONT_HERSHEY_PLAIN, 1.6, (255, 255, 255), 2)
        cv2.imwrite(str(dest / f'seam-z-vs-aa-{name}.png'), row)


def save_weights(dest):
    dest.mkdir(parents=True, exist_ok=True)
    wt = weights()
    z0 = None
    vis = np.zeros((H, W, 3), np.uint8)
    vis[..., 2] = (wt['head'] * 255).astype(np.uint8)
    vis[..., 1] = (wt['chest'] * 255).astype(np.uint8)
    vis[..., 0] = ((wt['ear_L'] + wt['ear_R']) * 200).astype(np.uint8)
    cv2.imwrite(str(dest / 'warp-weights.png'), vis)
    for k, m in wt.items():
        cv2.imwrite(str(dest / f'weight-{k}.png'), (np.clip(m, 0, 1) * 255).astype(np.uint8))


def main():
    import argparse
    global MOUTH_REF
    ap = argparse.ArgumentParser()
    ap.add_argument('--preview', action='store_true')
    ap.add_argument('--mouth-ref', default=None,
                    help='optional mouth-reference path (swap-in hook)')
    args = ap.parse_args()
    if args.mouth_ref:
        MOUTH_REF = args.mouth_ref

    mp4 = P / f'{VER}-test-25s.mp4'
    if not args.preview and mp4.exists():
        raise SystemExit(f'refuse to overwrite {mp4}')

    for label, path, expect in (
        ('v41u-ex-mp4', P / 'v41u-ex-test-25s.mp4', V41U_SHA),
        ('v41u-ex-los', SRC_U, V41U_LOS_SHA),
        ('v41z-ex-mp4', P / 'v41z-ex-test-25s.mp4', V41Z_SHA),
        ('v41z-ex-los', SRC_Z, V41Z_LOS_SHA),
        ('pcm', PCM, Hh.PCM_SHA),
    ):
        got = sha256(path)
        if got != expect:
            raise SystemExit(f'{label} sha changed: {got}')
    print('frozen shas ok', flush=True)
    print('mouth ref:', MOUTH_REF or '(none — hook is idle)', flush=True)

    print('loading plates…', flush=True)
    src_u = Hh.load_span(SRC_U)
    src_z = Hh.load_span(SRC_Z)
    g1 = Z.load_mp4(Z.GEN1)
    g2 = Z.load_mp4(Z.GEN2)
    a1 = np.load(Z.WORK / 'mattes' / 'gen1-final.npz')['alpha'].astype(np.float32) / 255.0
    a2 = np.load(Z.WORK / 'mattes' / 'gen2-final.npz')['alpha'].astype(np.float32) / 255.0

    WORK.mkdir(exist_ok=True)
    (WORK / 'mattes').mkdir(exist_ok=True)
    alpha = load_or_build_alpha(g1, g2, a1, a2, WORK / 'mattes' / 'z-alpha-600.npz')
    save_weights(QC / ('preview' if args.preview else 'encoded'))

    if args.preview:
        idx = sorted(set(
            list(range(0, 241, 10))
            + list(range(16, 50))
            + list(range(68, 82))
            + list(range(96, 130))
            + list(range(126, 155))
            + list(range(230, 252))
            + list(range(340, 360))
            + list(range(420, 450))
            + list(range(470, 500))
            + [599]
        ))
        print('preview frames', len(idx), flush=True)
        out = process(src_z, src_u, alpha, preview=idx)
        dest = QC / 'preview'
        Y.VER = VER
        Y.write_qc(out, src_u, dest, mp4=None)
        edge_qc(out, src_z, src_u, dest)
        breath_strip(out, src_z, dest)
        twitch_qc(out, src_z, dest)
        seam_report(out, src_z, dest)
        # flow on the preview-available adjacent pairs only (sparse is ok)
        print('computing preview flow (sparse)…', flush=True)
        # fill missing with z so the plot still has a z line everywhere
        mz = np.load('/tmp/v41aa-proto/z-flow-mags.npz')
        mags_z = {k: mz[k] for k in ('head', 'ears', 'chest', 'tail')}
        mags_aa = flow_mags(out, src_u)
        draw_flow_plot(mags_z, mags_aa, dest)
        print('preview written', dest, flush=True)
        return

    out = process(src_z, src_u, alpha)
    y0, y1, x0, x1 = TITLE
    bad = [i for i, (a, b) in enumerate(zip(src_z, out))
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
    shutil.copy2(mp4, WORK / f'{VER}-test-25s.mp4')
    shutil.copy2(Path(__file__), WORK / 'build_v41aa_ex.py')
    dest = QC / 'encoded'
    Y.VER = VER
    Y.write_qc(out, src_u, dest, mp4=mp4)
    edge_qc(out, src_z, src_u, dest)
    breath_strip(out, src_z, dest)
    twitch_qc(out, src_z, dest)
    seam_report(out, src_z, dest)
    print('computing flow mags…', flush=True)
    mz = flow_mags(src_z, src_u)
    ma = flow_mags(out, src_u)
    np.savez(dest / 'flow-mags-z-aa.npz', **{f'z_{k}': mz[k] for k in mz},
             **{f'aa_{k}': ma[k] for k in ma})
    draw_flow_plot(mz, ma, dest)
    # rest-hold summary
    def hold_frac(v, thr=0.15):
        return float(np.mean(v < thr))
    print('hold frac mag<0.15  z / aa')
    for k in ('head', 'ears', 'chest', 'tail'):
        print(f'  {k:6s}  {hold_frac(mz[k]):.3f}  {hold_frac(ma[k]):.3f}')


if __name__ == '__main__':
    main()
