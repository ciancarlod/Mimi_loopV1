#!/usr/bin/env python3
"""v41: fork of v40. Rigid eyes, clean lower-lid squint, less chin/chest stretch, smooth mrrrp.

Target: medium meows in the user green-screen clip
(/workspace/ref-greenscreen/greenscreen-meow-ref2.mp4), especially ~14.8–15.1s,
not the extreme 10s peak. Ranked fixes from the v38 review:

1. Inverted-V upper lip, 4–5 px raise at the philtrum at peak.
2. Jaw drop +3 px; corners/nearby lower cheek follow slightly; chin/neck stay solid.
3. Readable upper fangs (~2–3 px) plus tiny incisors.
4. Painterly volumetric tongue and cavity (warm room light).
5. 2–3 frame ease-in/out on jaw and lip; peaks stay synced to the meow.
6. Whisker pads +2 px up and out, subtle crease.
7. Stronger head tilt (~2.5°) and lower-lid squint +2–3 px.
8. Smoother lower-lip AA; choke the head mask ~1 px to kill the lift halo.
9. Chest rise +2–3 px.

Does not modify v30–v38, originals, masters or loops.
"""
import sys, cv2, numpy as np, json, subprocess as sp
from pathlib import Path

P = Path('/workspace/lofi-v41-meow')
G = Path('/workspace/lofi-v31-meow/gs')
N, PW, PH = 241, 116, 72
SCALE = 8
A, SV = 0.85, 0.25

def sstep(x):
    x = np.clip(x, 0, 1)
    return x * x * x * (x * (6 * x - 15) + 10)

v = cv2.VideoCapture('/workspace/lofi-v30/v30-repaired-cycle-lossless.mkv')
o = cv2.VideoCapture('/workspace/lofi-v24/v24-original-mimi-one-cycle-mild-squint.mp4')
v30, orig = [], []
for n in range(N):
    ok, f = v.read(); assert ok; v30.append(f)
    ok, f = o.read(); assert ok; orig.append(f)
v31p = json.load(open('/workspace/lofi-v31-meow/cycle/v31-mouth-params.json'))
active = v31p['active_frames']
BOX = {r['frame']: r['box'] for r in v31p['records']}

MEOWS = [(1.652, 2.38), (3.992, 4.74)]
LEAD, ATT = 0.27, 0.36

def curve(t):
    for a, e in MEOWS:
        t0 = a - LEAD
        D = e - a
        tr = a + 0.50 * D
        te = e + 0.05
        if t0 - 0.05 <= t <= te + 0.05:
            att = sstep((t - t0) / ATT)
            over = 0.035 * np.sin(np.pi * np.clip((t - (a + 0.08)) / 0.24, 0, 1)) ** 2
            wob = (0.012 * np.sin(2 * np.pi * 2.6 * (t - a))
                   * sstep((t - (a + 0.18)) / 0.1)
                   * (1 - sstep((t - tr + 0.05) / 0.1)))
            rel = 1 - sstep((t - tr) / (te - tr))
            p = float(np.clip(att * (1 + over + wob) * rel, 0, 1.05))
            closing = float(sstep((t - tr) / (te - tr)))
            gt = float(np.interp(t, [t0, a - 0.20, a + 0.02, tr, te],
                                 [9.96, 10.04, 10.16, 10.50, 10.82]))
            return p, closing, gt
    return 0., 0., None

gsx = np.array([60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 297])
gsy = np.array([335, 318, 300, 281, 263, 250, 236, 250, 259, 270, 288, 306, 322])
GJ, GSPAN = 180, (120, 117)

def y0g(x):
    return np.interp(x, gsx, gsy)

def p_geom(t):
    """5-tap (~2-frame) ease of p and closing; does not move the peak time.
    Near the peak, neighbours are also ~1 so the peak magnitude is preserved.
    At onset/release this averages with more-closed frames (ease-in/out)."""
    ks = (-2, -1, 0, 1, 2)
    ws = (1.0, 2.0, 3.0, 2.0, 1.0)
    ap = ac = sw = 0.0
    tg0 = None
    for k, w in zip(ks, ws):
        p, cl, tg = curve(t + k / 24.0)
        ap += w * p
        ac += w * cl
        sw += w
        if k == 0:
            tg0 = tg
    return ap / sw, ac / sw, tg0

mx = np.array([-32, -30, -24, -18, -14, -10, -6, -2, 0, 2, 4, 8, 12, 16, 20, 24, 30, 36, 40, 46, 52])
my = np.array([8, 8, 9, 9, 8, 6, 4, 2, 0, 0, 2, 6, 9, 11, 14, 16, 18, 20, 21, 22, 23]).astype(float)
PHIL = 4.0
PAD_OUT = 3.15  # v38 was 1.15; +2 px more outward at p=1 (item 6)
# Face midline in mouth-box x (philtrum groove / nose tip). Measured on
# rest frames 33 and 89 and open peaks 44/100/109: the dark philtrum shaft
# sits at box x ≈ 50 (the old warp peaked at x=48; painted PHIL=4 is x=52).
# Eyes midpoint is not used: the head is slightly turned so it lies right
# of the muzzle. Whisker-pad midpoint is ~53, close to this line.
MOUTH_MID = 50.0
CHIN_H = 18.0       # rigid chin/fur patch height below yL, src px
CHIN_FEATHER = 11.0  # alpha fade onto unmoved neck (no geometric stretch)
# Lift/drop units (scaled by SV*A*p). Chosen so that at p=1.033:
#   upper-lip V at philtrum ≈ 4.8 px (v38 was 17*sc*p ≈ 3.73 px)
#   jaw drop ≈ v38 24.59 + 3.0 = 27.59 px
LIFT_U = 21.9
DROP_U = 125.7
# Body language (eased with p). Head/chest are a soft rigid lift, not a stretchy warp.
HEAD_UP = 4.5     # was 3.0; plus muzzle tilt ≈ 2.5° nose-up
CHEST_UP = 4.0    # was 1.5; +2.5 px
SQUINT_UP = 1.5   # upper lid, barely more than v38
SQUINT_LO = 3.9   # lower lid: v38 1.4 + 2.5
WHISK_OUT = 1.3   # whisker-stroke push (pads have their own PAD_OUT)
PAD_UP = 2.0      # pads rise, src px at p=1
LIP_PART = 1.8    # stronger lateral split under the philtrum
CORNER_FOLLOW = 1.25  # local downward follow at commissures, src px
CREASE = 0.22     # subtle warm corner crease opacity at peak
# Per-occurrence style for the 25s span. Set by the renderer.
#   'medium'  open meow ~5.4s, 0:14/16-like slim oval + mild V
#   'mrrrp'   closed-mouth meow-growl ~7.7s, tiny lip flex + pulse
#   'growl'   closed-mouth meow-growl ~15.4s from ref4, lips sealed, tiny jaw/cheek
#   'big'     open meow ~17.75s, 0:10-like wide rounded diamond + corner swoop
STYLE = 'medium'
T_NOW = 0.0
# Shape scales (units * SV * A * p_geom). Slimmer than v39; big is still under v39 height.
SHAPES = {
    # Mouth opening units unchanged from v40 so the painted shapes stay the same.
    # Head/chest nearly uniform (was a 0.8–1.0 px split that flattened fur).
    # squint_up is unused: only a clean 1–2 px lower-lid rise, iris held rigid.
    'medium': dict(LIFT_U=11.2, DROP_U=63.0, drop_exp=2.4, lift_exp=1.20, gspan=(120, 117),
                   head=2.8, chest=2.6, tilt=1.0, squint_lo=1.6, squint_up=0.0,
                   pad_out=2.4, pad_up=1.4, lip_part=1.3, corner=0.8, crease=0.16,
                   teeth='upper', commissure=0.25),
    'big':    dict(LIFT_U=14.0, DROP_U=71.4, drop_exp=3.2, lift_exp=1.15, gspan=(130, 127),
                   head=3.8, chest=3.6, tilt=1.8, squint_lo=2.0, squint_up=0.0,
                   pad_out=3.4, pad_up=2.2, lip_part=2.0, corner=1.4, crease=0.28,
                   teeth='four', commissure=0.85),
    'mrrrp':  dict(LIFT_U=1.7, DROP_U=3.5, drop_exp=2.6, lift_exp=1.4, gspan=(120, 117),
                   head=1.0, chest=1.0, tilt=0.2, squint_lo=1.4, squint_up=0.0,
                   pad_out=1.6, pad_up=0.7, lip_part=0.1, corner=0.3, crease=0.16,
                   teeth='none', commissure=0.0),
    'growl':  dict(LIFT_U=0.0, DROP_U=0.0, drop_exp=2.0, lift_exp=1.0, gspan=(120, 117),
                   head=0.8, chest=0.8, tilt=0.2, squint_lo=1.3, squint_up=0.0,
                   pad_out=1.8, pad_up=0.9, lip_part=0.0, corner=0.25, crease=0.14,
                   teeth='none', commissure=0.0, jaw_down=1.3),
}

def shape():
    return SHAPES[STYLE]


def rest_crease_old(dx48):
    """v34 painted rest crease, dx measured from box x=48."""
    return np.interp(dx48, mx, my)

def rest_crease_sym(dx_mid):
    """Lower lip symmetric about MOUTH_MID: left-side profile mirrored
    to the right, so the right droop is lifted to match the left."""
    # old table is keyed from box x=48; mid is 2 px to the right of that.
    return np.interp((MOUTH_MID - 48.0) - np.abs(dx_mid), mx, my)
S = {g: cv2.imread(str(G / f'stab/s{g}.png')).astype(np.float32) for g in range(246, 277)}
GSCHIN = np.array([124.7, 134.8, 158.4])
MCHIN = np.array([91., 155., 192.])

def _lab(c):
    return cv2.cvtColor(np.uint8([[c]]), cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)

LG, LM = _lab(GSCHIN), _lab(MCHIN)

def colorize(t):
    lab = cv2.cvtColor(np.clip(t, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[..., 0] *= LM[0] / LG[0]
    lab[..., 1] += (LM[1] - LG[1]) * .8
    lab[..., 2] += (LM[2] - LG[2]) * .8
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)

def profiles(u, p, closing):
    au = np.abs(u)
    sh = shape()
    # Less side lag than v34/v39 so the opening can fill out, but corners still ease.
    k = 0.55 + 1.3 * (1 - min(p, 1)) ** 2 * (1 - closing) + 1.6 * closing ** 1.5
    px = np.where(au < 1, p ** (1 + k * u * u), 0)
    lift = sh['LIFT_U'] * np.clip(1.0 - np.abs(u) ** sh['lift_exp'], 0, 1) * px
    drop = sh['DROP_U'] * np.clip(1.0 - np.minimum(au, 1) ** sh['drop_exp'], 0, 1) * px
    return lift, drop

yy, xx = np.mgrid[0:PH, 0:PW].astype(np.float32)
Y8, X8 = np.mgrid[0:PH * SCALE, 0:PW * SCALE].astype(np.float32)
X8 /= SCALE
Y8 /= SCALE

def ramp(d, a, b):
    return sstep((d - a) / (b - a))

EW = np.minimum.reduce([
    ramp(xx, 3, 16), ramp(PW - 1 - xx, 3, 16),
    ramp(yy, 3, 10), ramp(PH - 1 - yy, 3, 12)
])
EW8 = np.minimum.reduce([
    ramp(X8, 3, 16), ramp(PW - 1 - X8, 3, 16),
    ramp(Y8, 3, 10), ramp(PH - 1 - Y8, 3, 12)
])

def down_area(img, scale=SCALE):
    h, w = img.shape[:2]
    if img.ndim == 2:
        return img.reshape(h // scale, scale, w // scale, scale).mean((1, 3)).astype(np.float32)
    return img.reshape(h // scale, scale, w // scale, scale, img.shape[2]).mean((1, 3)).astype(np.float32)

def soft_inside(dy, yU, yL, u, aa_up=0.95, aa_lo=1.20):
    """Analytic coverage. Slightly wider AA on the lower lip to ease 1 px stepping,
    without softening the upper lip into mush."""
    v_up = sstep(((dy - yU) + aa_up) / (2 * aa_up))
    v_lo = sstep(((yL - dy) + aa_lo) / (2 * aa_lo))
    hcov = sstep((1.0 - np.abs(u) + 0.035) / (2 * 0.035))
    return np.clip(v_up * v_lo * hcov, 0, 1)

def paint_teeth(tex, dx, dy, yU, yL, p, u):
    """Anti-aliased canines (and incisors) in 8x src-px space."""
    mode = shape().get('teeth', 'upper')
    if mode == 'none' or p < 0.08:
        return tex
    amp = sstep(np.clip((p - 0.08) / 0.25, 0, 1))
    # Canines hang just under the upper lip, slightly off the philtrum.
    fangs = np.zeros(dx.shape, np.float32)
    shade = np.zeros(dx.shape, np.float32)  # 0=lit, 1=shadow side
    for cx, side in ((-6.6, 0.22), (6.6, 0.72)):  # right fang a bit more shaded (lamp is left)
        fx = dx - cx
        # teardrop / rounded triangle; TOOTH_S=0.70 vs v40 so fangs fit the
        # ~30% smaller opening and stay readable (~2.0 px tall, ~0.84 px half).
        TOOTH_S = 0.70
        fy = (dy - (yU + 0.22)) / (2.85 * TOOTH_S)
        half = 1.20 * TOOTH_S * np.clip(1.0 - fy * 0.85, 0.15, 1.0)
        wx = np.abs(fx) / np.maximum(half, 0.15)
        body = (1.0 - sstep((wx - 1.0) / 0.22)) * (1.0 - sstep((fy - 1.0) / 0.12)) * sstep((fy + 0.12) / 0.18)
        body = body * (1.0 - sstep((np.abs(u) - 0.72) / 0.12))
        fangs = np.maximum(fangs, body.astype(np.float32))
        # self-shadow: right side of each tooth + overall side
        sh = sstep((fx / np.maximum(half, 0.15) + 0.15) / 0.85)
        shade = np.where(body > fangs * 0.5, np.maximum(shade, 0.35 * sh + side * 0.25), shade)
    # Tiny incisors between the canines, ~1.0 px tall
    inc = np.zeros(dx.shape, np.float32)
    for cx in (-2.0, 0.0, 2.0):
        fx = dx - cx
        fy = (dy - (yU + 0.22)) / (1.05 * 0.70)
        wx = np.abs(fx) / (0.55 * 0.70)
        body = (1.0 - sstep((wx - 1.0) / 0.25)) * (1.0 - sstep((fy - 1.0) / 0.15)) * sstep((fy + 0.15) / 0.2)
        inc = np.maximum(inc, body.astype(np.float32))
    fangs = np.clip(cv2.GaussianBlur(fangs, (0, 0), 0.6), 0, 1) * amp
    inc = np.clip(cv2.GaussianBlur(inc, (0, 0), 0.5), 0, 1) * amp * 0.85
    shade = cv2.GaussianBlur(shade.astype(np.float32), (0, 0), 0.6)
    lit = np.array([228., 238., 252.], np.float32)   # warm white, BGR (lamp)
    shc = np.array([170., 196., 220.], np.float32)
    inc_c = np.array([200., 214., 230.], np.float32)
    fang_c = lit * (1.0 - shade[..., None]) + shc * shade[..., None]
    w = fangs[..., None]
    tex = tex * (1.0 - w) + fang_c * w
    w2 = inc[..., None] * (1.0 - w)
    tex = tex * (1.0 - w2) + inc_c * w2
    if mode == 'four' and p > 0.28:
        # Lower canines sit on the lower lip, slightly more lateral than the uppers.
        amp_l = sstep(np.clip((p - 0.28) / 0.25, 0, 1))
        low = np.zeros(dx.shape, np.float32)
        lsh = np.zeros(dx.shape, np.float32)
        for cx, side in ((-9.0, 0.25), (9.0, 0.70)):
            fx = dx - cx
            fy = ((yL - 0.20) - dy) / (2.35 * 0.70)
            half = 1.10 * 0.70 * np.clip(1.0 - fy * 0.85, 0.15, 1.0)
            wx = np.abs(fx) / np.maximum(half, 0.15)
            body = (1.0 - sstep((wx - 1.0) / 0.22)) * (1.0 - sstep((fy - 1.0) / 0.12)) * sstep((fy + 0.12) / 0.18)
            body = body * (1.0 - sstep((np.abs(u) - 0.78) / 0.10))
            low = np.maximum(low, body.astype(np.float32))
            sh = sstep((fx / np.maximum(half, 0.15) + 0.15) / 0.85)
            lsh = np.where(body > low * 0.5, np.maximum(lsh, 0.35 * sh + side * 0.25), lsh)
        low = np.clip(cv2.GaussianBlur(low, (0, 0), 0.6), 0, 1) * amp_l
        lsh = cv2.GaussianBlur(lsh.astype(np.float32), (0, 0), 0.6)
        fang_c = lit * (1.0 - lsh[..., None]) + shc * lsh[..., None]
        w3 = low[..., None]
        tex = tex * (1.0 - w3) + fang_c * w3
    return tex


def render_one(n, p, closing, tg, base_fur, y_off=0):
    x0, y0, _, _ = BOX[n]
    y0 = y0 + y_off
    cx, cy = x0 + 48.0, y0 + 20.0
    # Geometry is now centered on the measured philtrum (box x=MOUTH_MID),
    # not the old u=0 at x=48.
    dx = X8 - MOUTH_MID
    dy = Y8 - 20
    sc = SV * A
    gs = shape()['gspan']
    xg = GJ + dx / sc
    u = np.where(xg < GJ, (xg - GJ) / gs[0], (xg - GJ) / gs[1])
    l_old = rest_crease_old(X8 - 48.0)
    l_sym = rest_crease_sym(dx)
    # Full symmetry by early-open (p~0.35) so the droop fix is visible at
    # mid and peak; at p~0 we still follow the painted crease (and skip).
    amt = sstep(np.clip(p / 0.35, 0, 1))
    l0 = (1.0 - amt) * l_old + amt * l_sym
    lg, dg = profiles(u, p, closing)
    lift = sc * lg
    drop = sc * dg
    yU = l0 - lift
    yL = l0 + drop
    # Upper lip: same local pull as v34. Lower face: NO cosine Y-falloff.
    wu = np.exp(-(np.maximum(yU - dy, 0) / 6.) ** 2)
    disp_up = np.where(dy <= yU, -lift * wu, 0.) * EW8
    sy_base = dy - disp_up
    sy_base = np.where((dy > yU) & (dy < yL), l0, sy_base)
    dx_ph = dx
    def _pad(cx, cy_, sx, sy):
        return np.exp(-0.5 * ((dx_ph - cx) / sx) ** 2 - 0.5 * ((dy - cy_) / sy) ** 2)
    wl = _pad(-22.0, 3.0, 11.0, 10.0)
    wr = _pad( 22.0, 3.0, 11.0, 10.0)
    outside = np.clip(sstep((0.5 - np.minimum(dy - yU, yL - dy)) / 1.2), 0, 1)
    disp_x = (wl * (-shape()['pad_out']) + wr * (shape()['pad_out'])) * p * A * outside * EW8
    disp_y_pad = (wl + wr) * shape()['pad_up'] * p * A * outside * EW8
    part_w = (np.exp(-0.5 * (dx / 4.5) ** 2)
              * np.exp(-0.5 * ((dy - yU + 1.5) / 4.0) ** 2)
              * (1.0 - sstep((dy - yU) / 1.2)))
    disp_x = disp_x + np.tanh(dx / 2.0) * shape()['lip_part'] * min(p, 1.0) * A * part_w * EW8
    corner = (np.exp(-0.5 * ((dx + 22.0) / 7.0) ** 2) + np.exp(-0.5 * ((dx - 22.0) / 7.0) ** 2))
    corner = corner * np.exp(-0.5 * ((dy - l0 - 2.0) / 6.5) ** 2)
    corner *= (1.0 - sstep((dy - l0 - 8.0) / 4.0))
    corner_d = shape()['corner'] * min(p, 1.0) * A * corner * EW8
    disp_x = disp_x + np.tanh(dx / 16.0) * 0.30 * corner_d
    sy_base = sy_base - corner_d
    # Rigid chin/fur patch: one mid-line translation, no EW taper on the
    # displacement (that taper was a stretch). Feathered alpha only.
    midc = int(np.clip(round(MOUTH_MID * SCALE), 0, drop.shape[1] - 1))
    drop_mid = float(drop[drop.shape[0] // 2, midc])
    sy_chin = dy - drop_mid
    src_x = (X8 + x0 - disp_x).astype(np.float32)
    fur_base = cv2.remap(base_fur, src_x, (sy_base + cy + disp_y_pad).astype(np.float32),
                         interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    src_x_chin = (X8 + x0).astype(np.float32)
    fur_chin = cv2.remap(base_fur, src_x_chin, (sy_chin + cy).astype(np.float32),
                         interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    dchin = dy - yL
    chin_a = sstep((dchin + 0.8) / 1.2)
    chin_a = chin_a * (1.0 - sstep((dchin - (CHIN_H - CHIN_FEATHER)) / max(CHIN_FEATHER, 1e-3)))
    chin_a = np.where(dchin < -0.5, 0.0, chin_a)
    chin_a = chin_a * EW8
    chin_a = np.clip(chin_a, 0, 1).astype(np.float32)
    fur = fur_base * (1.0 - chin_a[..., None]) + fur_chin * chin_a[..., None]
    # Subtle warm crease at each corner (not a black sharpie line).
    if p > 1e-4:
        crease = (np.exp(-0.5 * ((dx + 22.0) / 2.2) ** 2) + np.exp(-0.5 * ((dx - 22.0) / 2.2) ** 2))
        crease = crease * np.exp(-0.5 * ((dy - l0 - 1.0) / 3.0) ** 2)
        crease = np.clip(crease * shape()['crease'] * min(p, 1.0) * A, 0, 1)
        crease = cv2.GaussianBlur(crease.astype(np.float32), (0, 0), 1.8)
        warm = np.array([58., 92., 128.], np.float32)
        fur = fur * (1.0 - 0.55 * crease[..., None]) + warm * (0.55 * crease[..., None])
    inside = np.zeros((PH * SCALE, PW * SCALE), np.float32)
    tex = np.zeros_like(fur)
    if p > 1e-4:
        f = np.clip((dy - yU) / np.maximum(yL - yU, 1e-3), 0, 1)
        yUg = y0g(xg) - lg
        yLg = y0g(xg) + dg
        ygs = yUg + (0.02 + 0.965 * f) * (yLg - yUg)
        k = tg * 25
        k0 = int(np.floor(k))
        al = k - k0
        k0 = min(max(k0, 246), 275)
        t0 = cv2.remap(S[k0], xg.astype(np.float32), ygs.astype(np.float32),
                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        t1 = cv2.remap(S[k0 + 1], xg.astype(np.float32), ygs.astype(np.float32),
                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        tex = colorize((1 - al) * t0 + al * t1)
        g = tex.mean(2, keepdims=True)
        tex = (g + (tex - g) * .85) * (0.80 + 0.12 * f[..., None])
        gap = np.maximum(yL - yU, 0)
        # Visible-opening gate: a thin gap keeps the natural crease instead of
        # a dark-blue slit. Cavity/shade fade in only once the mouth is open.
        cav = np.clip((gap - 3.5) / 7.0, 0, 1)
        cav = cav * cav * (3.0 - 2.0 * cav)
        src_tex = tex.copy()
        au = np.abs(u)
        # v34's pale "human teeth" read is the bright LOWER slab, not the
        # upper canines. Leave the upper cavity (and fangs) close to v34;
        # recess the mid mouth and replace the lower slab with a muted tongue.
        Y0 = src_tex.mean(2)
        au = np.abs(u)
        WARM_DARK = np.array([26., 32., 48.], np.float32)   # throat, warm room shadow
        WARM_MID  = np.array([40., 50., 70.], np.float32)
        TONGUE    = np.array([70., 96., 150.], np.float32)
        TONGUE_LIT= np.array([100., 128., 176.], np.float32)
        TONGUE_EDGE=np.array([50., 68., 110.], np.float32)
        # Throat: dark recess in the mid/upper cavity, darker toward the sides.
        throat = (sstep((f - 0.06) / 0.10) * (1.0 - sstep((f - 0.48) / 0.16)))
        throat = throat * (0.65 + 0.35 * sstep((au - 0.15) / 0.50))
        tex = tex * (1.0 - 0.70 * (throat * cav)[..., None]) + WARM_DARK * (0.70 * (throat * cav)[..., None])
        mid = (sstep((f - 0.20) / 0.12) * (1.0 - sstep((f - 0.58) / 0.14))
               * (1.0 - sstep((au - 0.80) / 0.16)))
        tex = tex * (1.0 - 0.35 * mid[..., None]) + WARM_MID * (0.35 * mid[..., None])
        # Tongue at the bottom: volumetric — darker at the root/edges, a soft
        # warm highlight on the upper face of the tongue (room lamp).
        tw = sstep((f - 0.42) / 0.12) * (1.0 - sstep((f - 0.97) / 0.05)) * (1.0 - sstep((au - 0.80) / 0.16))
        edge = sstep((au - 0.35) / 0.40)
        root = 1.0 - sstep((f - 0.52) / 0.18)  # toward the throat
        hl = sstep((0.78 - f) / 0.16) * sstep((f - 0.50) / 0.10) * (1.0 - sstep((au - 0.45) / 0.30))
        # bias highlight to the left (lamp)
        hl = hl * (0.75 + 0.25 * sstep((0.15 - dx) / 8.0))
        tcol = TONGUE * (1.0 - 0.45 * root[..., None] - 0.35 * edge[..., None] + 0.0)
        tcol = tcol * (1.0 - 0.40 * edge[..., None]) + TONGUE_EDGE * (0.40 * edge[..., None])
        tcol = tcol * (1.0 - 0.55 * hl[..., None]) + TONGUE_LIT * (0.55 * hl[..., None])
        tex = tex * (1.0 - 0.86 * tw[..., None]) + tcol * (0.86 * tw[..., None])
        # Soft interior shade only — never a hard black outline on the lip.
        shade = np.clip(1 - (gap - 2.5) / 10.0, 0, 1)
        shade = np.clip(shade + 0.10 * closing * np.clip(1 - (gap - 5.0) / 12.0, 0, 1), 0, 1)
        warm_shade = np.array([58., 74., 102.], np.float32)
        shade = shade * cav
        tex = tex * (1 - 0.40 * shade[..., None]) + warm_shade * (0.40 * shade[..., None])
        tex = paint_teeth(tex, dx, dy, yU, yL, p, u)
        # Tapered commissure line + corner shadow (stronger on the big 0:10-like meow).
        com = shape().get('commissure', 0.0)
        if com > 0 and p > 0.12:
            au = np.abs(u)
            # A dark stroke that runs out toward the pad corners along the lower lip,
            # tapering (cat-like) rather than a hard slot.
            line = np.exp(-0.5 * ((dy - yL) / 1.15) ** 2)
            span = sstep((au - 0.42) / 0.18) * (1.0 - sstep((au - 1.12) / 0.10))
            # Cavity itself also darkens toward the corners.
            cav_edge = sstep((au - 0.55) / 0.30) * sstep((0.96 - au) / 0.12)
            darkc = np.array([22., 28., 44.], np.float32)
            wline = np.clip(line * span * com * min(p, 1.0) * cav, 0, 1)
            wline = cv2.GaussianBlur(wline.astype(np.float32), (0, 0), 1.2)
            tex = tex * (1.0 - 0.45 * wline[..., None]) + darkc * (0.45 * wline[..., None])
            wcav = np.clip(cav_edge * 0.55 * com * min(p, 1.0), 0, 1)
            tex = tex * (1.0 - wcav[..., None]) + darkc * wcav[..., None]
        crease_col = fur * 0.65 + np.array([80., 100., 130.], np.float32) * 0.35
        tex = tex * cav[..., None] + crease_col * (1.0 - cav)[..., None]
        inside = soft_inside(dy, yU, yL, u, aa_up=1.15, aa_lo=1.40)
        inside = cv2.GaussianBlur(inside, (0, 0), 5.5)
        inside = np.clip(inside, 0, 1)
        # Choke the dark cavity so it stays inside the lip; the outer band is
        # colour-matched to the surrounding fur (no hard dark cut-out).
        core = np.clip((inside - 0.55) / 0.35, 0, 1)
        core = core * core * (3.0 - 2.0 * core)
        edge = np.clip(inside - core, 0, 1)
        tex = cv2.GaussianBlur(tex, (0, 0), 4.0)
        edge_col = fur * 0.72 + tex * 0.28
        tex_comp = fur * (1.0 - inside)[..., None] + tex * core[..., None] + edge_col * edge[..., None]
        tex = tex_comp
        inside = inside
    if p > 1e-4:
        comp = tex
    else:
        comp = fur
    return down_area(comp)


# ---------- head / chest / eyes / whiskers (meow body language) ----------

def _ebox(n):
    x0, y0, w, h = BOX[n]
    # Stay right of the mug-steam (x~451), below the title card, and left of
    # the right ear / mic. Ears themselves are excluded by the top-corner taper.
    xa = max(470, x0 - 62)
    ya = max(220, y0 - 102)
    xb = min(875, x0 + w + 72)
    yb = min(595, y0 + h + 98)
    return [int(xa), int(ya), int(xb - xa), int(yb - ya)]


def _smooth_eyes():
    """Stable eye centres for every active frame, from v30, with group medians."""
    def detect(n, box):
        x0, y0, w, h = box
        xa, ya = max(0, x0 - 50), max(0, y0 - 115)
        xb, yb = min(1280, x0 + w + 60), y0 + 8
        roi = v30[n][ya:yb, xa:xb]
        Y = cv2.cvtColor(roi, cv2.COLOR_BGR2YUV)[:, :, 0].astype(np.float32)
        blur = cv2.GaussianBlur(Y, (11, 11), 0)
        dark = ((Y < blur - 8) & (Y < 90)).astype(np.uint8)
        dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        nlab, _, stats, cents = cv2.connectedComponentsWithStats(dark, 8)
        cands = []
        for i in range(1, nlab):
            x, y, ww, hh, area = stats[i]
            if area < 80 or area > 2500 or ww < 12 or hh < 6 or ww > 80 or hh > 50:
                continue
            cx, cy = cents[i]
            if cy < 8 or cy > roi.shape[0] - 10:
                continue
            cands.append((area, float(cx + xa), float(cy + ya)))
        cands.sort(reverse=True)
        top = cands[:8]
        best = None
        for i in range(len(top)):
            for j in range(i + 1, len(top)):
                dx = abs(top[i][1] - top[j][1]); dy = abs(top[i][2] - top[j][2])
                if dx < 40 or dx > 160 or dy > 40:
                    continue
                pair = sorted([top[i], top[j]], key=lambda t: t[1])
                score = dx * (min(pair[0][0], pair[1][0]) ** 0.5)
                if best is None or score > best[0]:
                    best = (score, pair)
        if not best:
            return None
        L, R = best[1]
        return (L[1], L[2], R[1], R[2])

    out = {}
    for group in (list(range(34, 63)), list(range(90, 119))):
        raw = {}
        for n in group:
            raw[n] = detect(n, BOX[n])
        goods = [raw[n] for n in group if raw[n] is not None]
        # reject y-outliers vs median
        if goods:
            myL = np.median([g[1] for g in goods]); myR = np.median([g[3] for g in goods])
            mxL = np.median([g[0] for g in goods]); mxR = np.median([g[2] for g in goods])
            cleaned = []
            for n in group:
                g = raw[n]
                if g is None or abs(g[1] - myL) > 12 or abs(g[3] - myR) > 12:
                    cleaned.append(None)
                else:
                    cleaned.append(g)
            # fill + slight temporal smooth
            filled = []
            last = (mxL, myL, mxR, myR)
            for g in cleaned:
                last = g if g is not None else last
                filled.append(last)
            # backward fill start
            first = next(i for i, g in enumerate(filled) if cleaned[i] is not None) if any(cleaned) else 0
            for i in range(first):
                filled[i] = filled[first]
            arr = np.array(filled, np.float32)
            k = 3
            pad = np.pad(arr, ((k//2, k//2), (0, 0)), mode='edge')
            sm = np.stack([pad[i:i + len(arr)].mean(0) for i in range(k)], 0).mean(0) if False else arr
            # simple 3-tap
            sm = arr.copy()
            for i in range(len(arr)):
                sl = arr[max(0, i - 1):min(len(arr), i + 2)]
                sm[i] = sl.mean(0)
            for n, g in zip(group, sm):
                # typical almond radii (src px); right eye is a touch wider
                # Generous almond radii so corners and the lower pupil stay inside
                # the rigid globe (v40's 17x10 / 19x10.5 left both leaking).
                out[n] = ((float(g[0]), float(g[1]), 22.0, 13.0),
                          (float(g[2]), float(g[3]), 26.0, 13.5))
    return out

EYES = _smooth_eyes()


def cat_mask(img, ebox, n=None):
    xa, ya, w, h = ebox
    roi = img[ya:ya + h, xa:xa + w]
    b, g, r = cv2.split(roi.astype(np.float32))
    Y = cv2.cvtColor(roi, cv2.COLOR_BGR2YUV)[:, :, 0].astype(np.float32)
    # Warm fur only. Cool window snow/sky is grey-blue (B >= R) and must stay out.
    orange = (r > g + 8) & (r > b + 20) & (r > 90)
    white = (Y > 145) & (r >= b + 2) & (g >= b - 6)
    pink = (r > g + 5) & (r > b + 8) & (r > 80) & (r < 210) & (Y > 60) & (Y < 200)
    raw = (orange | white | pink).astype(np.uint8)
    raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    # Fill only small interior holes (eyes, nose, mouth). Do NOT fill large
    # background pockets such as the window, which can be enclosed by the box.
    body = raw.copy()
    nlab, lab, stats, _ = cv2.connectedComponentsWithStats((raw == 0).astype(np.uint8), 8)
    for i in range(1, nlab):
        x, y, ww, hh, area = stats[i]
        touches = x <= 1 or y <= 1 or (x + ww) >= w - 2 or (y + hh) >= h - 2
        if (area < 2500) and not touches:
            body[lab == i] = 1
    body = cv2.morphologyEx(body, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    def _ramp(d, a, b):
        return sstep((d - a) / max(b - a, 1e-6))
    ew = np.minimum.reduce([
        _ramp(xx, 2, 18), _ramp(w - 1 - xx, 2, 18),
        _ramp(yy, 2, 22), _ramp(h - 1 - yy, 2, 22),
    ])
    # Taper the top so ear corners stay outside the active region.
    yc = np.clip(yy / max(0.45 * h, 1), 0, 1)
    half = (0.20 + 0.30 * sstep(yc)) * w
    taper = sstep((half - np.abs(xx - w / 2.0)) / 14.0)
    m = body.astype(np.float32) * ew * taper
    if n is not None:
        x0, y0, _, _ = BOX[n]
        cx = (x0 + MOUTH_MID) - xa
        cy = (y0 + 8) - ya
        prior = 1.0 - sstep((((xx - cx) / 118.0) ** 2 + ((yy - cy) / 150.0) ** 2 - 1.0) / 0.28)
        m = m * prior
    m = cv2.GaussianBlur(m, (0, 0), 3.0)
    # Choke ~1 px and slightly harden the edge so the head lift does not
    # leave a faint halo against the unmoved background.
    m = cv2.erode(m, np.ones((3, 3), np.uint8), iterations=1)
    m = np.clip(m * 1.05, 0, 1)
    return m.astype(np.float32)



def eye_masks(h, w, xa, ya, n):
    """Soft masks in ebox space.
    globe: iris, pupil, catchlights, eye outline (must stay rigid).
    brow: forehead/brow above the globe (exclude from pad/muzzle/tilt stretch).
    lid: lower-lid band just below the globe (the only place a squint warp is allowed).
    """
    globe = np.zeros((h, w), np.float32)
    brow_m = np.zeros((h, w), np.float32)
    lid = np.zeros((h, w), np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    if n not in EYES:
        return globe, lid, brow_m
    for (ecx, ecy, erx, ery) in EYES[n]:
        ex = xx - (ecx - xa)
        ey = yy - (ecy - ya)
        rx, ry = erx, ery
        te = (ex / max(rx, 1e-3)) ** 2 + (ey / max(ry, 1e-3)) ** 2
        g = (1.0 - sstep((te - 1.00) / 0.18)).astype(np.float32)
        globe = np.maximum(globe, g)
        # Brow / forehead lobe, ending at the upper lid so we don't eat the lower lid band.
        teb = (ex / (rx * 1.20)) ** 2 + ((ey + 10.0) / (ry + 22.0)) ** 2
        b = (1.0 - sstep((teb - 1.0) / 0.22)) * (1.0 - sstep((ey - 1.0) / 3.0))
        brow_m = np.maximum(brow_m, b.astype(np.float32))
        # Lower lid only: starts at the lower outline, dies ~7 px below, dies laterally
        # with the eye, and is forced to 0 inside the globe so the iris cannot move.
        below = sstep((ey - ry * 0.70) / 1.8) * (1.0 - sstep((ey - (ry + 7.0)) / 2.5))
        lateral = 1.0 - sstep((np.abs(ex) - rx * 1.05) / 3.5)
        lid = np.maximum(lid, (below * lateral * (1.0 - g)).astype(np.float32))
    return globe, lid, brow_m


def growl_curve(t, lead=0.27, att=0.50):
    """Slow, jitter-free closed-mouth envelope. Same time window as the meow
    stem (so growl audio stays in sync) but no overshoot and no 2.6 Hz wobble.
    Release eases all the way to 0 over ~4 frames; there is no snap."""
    for a, e in MEOWS:
        t0 = a - lead
        D = e - a
        tr = a + 0.50 * D
        te = e + 0.20
        if t0 - 0.08 <= t <= te + 0.08:
            att_p = sstep((t - t0) / att)
            rel = 1.0 - sstep((t - tr) / max(te - tr, 1e-3))
            p = float(np.clip(att_p * rel, 0, 1.0))
            closing = float(sstep((t - tr) / max(te - tr, 1e-3)))
            return p, closing, None
    return 0., 0., None


def apply_body(out, n, p):
    """Head/chest lift, whisker/pad push, and a clean lower-lid squint.

    The eye globes, outlines and brow never take a non-rigid warp. They
    translate only with the rigid head lift. Squint is a 1–2 px rise of the
    lower lid that is masked out of the iris/pupil/catchlight.
    """
    q = float(sstep(np.clip(p, 0, 1)))
    if q <= 1e-4:
        return out, _ebox(n), 0.0
    ebox = _ebox(n)
    xa, ya, w, h = ebox
    roi = out[ya:ya + h, xa:xa + w].astype(np.float32)
    mask = cat_mask(out, ebox, n)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    fy = yy / max(h, 1)
    sh = shape()
    # Nearly-uniform lift from crown through mid-chest (tiny head/chest delta)
    # so the chest fur does not flatten. Tilt is a muzzle-only bump, not an
    # eye-region gradient (sigma tightened, centre dropped toward the nose).
    head_w = 1.0 - sstep((fy - 0.58) / 0.16)
    chest_w = sstep((fy - 0.58) / 0.16) * (1.0 - sstep((fy - 0.86) / 0.12))
    tilt = sh['tilt'] * np.exp(-0.5 * ((fy - 0.54) / 0.10) ** 2)
    lift = q * (head_w * sh['head'] + chest_w * sh['chest'] + tilt)
    pulse = 0.0
    if sh.get('pulse_hz'):
        pulse = 0.5 * (1.0 + np.sin(2.0 * np.pi * sh['pulse_hz'] * T_NOW))
        x0b, y0b, _, _ = BOX[n]
        mouth_fy = np.clip(((y0b + 26) - ya) / max(h, 1), 0.2, 0.9)
        throat = sstep((fy - (mouth_fy + 0.02)) / 0.08) * (1.0 - sstep((fy - 0.90) / 0.08))
        lift = lift + q * sh['pulse_amp'] * pulse * throat
    if STYLE in ('growl', 'mrrrp'):
        x0b, y0b, _, _ = BOX[n]
        mid_fy = np.clip(((y0b + 20) - ya) / max(h, 1), 0.2, 0.8)
        lip_band = np.exp(-0.5 * ((fy - mid_fy) / 0.10) ** 2)
        lift_lip = q * ((1.0 - sstep((mid_fy - 0.56) / 0.14)) * sh['head']
                        + (sstep((mid_fy - 0.56) / 0.14) * (1.0 - sstep((mid_fy - 0.82) / 0.14))) * sh['chest']
                        + sh['tilt'] * np.exp(-0.5 * ((mid_fy - 0.54) / 0.10) ** 2))
        lift = lift * (1.0 - lip_band) + lift_lip * lip_band
    x0, y0, _, _ = BOX[n]
    mid_x = (x0 + MOUTH_MID) - xa
    pad_y = (y0 + 23) - ya
    wx = xx - mid_x
    wy = yy - pad_y
    cone = (sstep((np.abs(wx) - 16) / 8.0)
            * (1.0 - sstep((np.abs(wx) - 88) / 12.0))
            * np.exp(-0.5 * (wy / 22.0) ** 2))
    whisk = q * WHISK_OUT * cone
    globe, lid, brow_m = eye_masks(h, w, xa, ya, n)
    protect = np.clip(np.maximum(globe, brow_m), 0, 1)
    # Rigid lift used on the eye/brow: the value at each eye centre (head plate).
    rigid_lift = np.zeros((h, w), np.float32)
    if n in EYES:
        acc = np.zeros((h, w), np.float32)
        wgt = np.zeros((h, w), np.float32)
        for (ecx, ecy, erx, ery) in EYES[n]:
            exi = int(np.clip(round(ecx - xa), 0, w - 1))
            eyi = int(np.clip(round(ecy - ya), 0, h - 1))
            L = float(lift[eyi, exi])
            w_i = np.exp(-0.5 * (((xx - (ecx - xa)) / 40.0) ** 2 + ((yy - (ecy - ya)) / 36.0) ** 2))
            acc += w_i * L
            wgt += w_i
        rigid_lift = acc / np.maximum(wgt, 1e-6)
    else:
        rigid_lift = lift.copy()

    src_x = (xx + xa) - np.sign(wx + 1e-6) * whisk
    if STYLE in ('mrrrp', 'growl'):
        padw = np.exp(-0.5 * ((np.abs(wx) - 24) / 9.0) ** 2) * np.exp(-0.5 * (wy / 11.0) ** 2)
        padw = padw * sstep((np.abs(wx) - 12.0) / 4.0)
        # Never let the pad field reach the eyes or brow.
        padw = padw * (1.0 - protect)
        src_x = src_x - np.sign(wx + 1e-6) * q * (sh['pad_out'] + 0.55 * pulse * sh.get('pulse_amp', 0)) * padw
        lift = lift + q * (sh['pad_up'] + 0.40 * pulse * sh.get('pulse_amp', 0)) * padw
    if STYLE == 'growl' and sh.get('jaw_down'):
        # Tiny downward jaw on the chin only, starting well below the sealed crease.
        x0b, y0b, _, _ = BOX[n]
        mouth_fy = np.clip(((y0b + 32) - ya) / max(h, 1), 0.2, 0.9)
        jaw = q * float(sh['jaw_down']) * sstep((fy - (mouth_fy + 0.04)) / 0.06) * (1.0 - sstep((fy - 0.82) / 0.10))
        lift = lift - jaw
    src_y = (yy + ya) + lift
    # Eyes + brow: identity in x, rigid head lift in y. No squint here.
    src_x = src_x * (1.0 - protect) + (xx + xa) * protect
    src_y = src_y * (1.0 - protect) + ((yy + ya) + rigid_lift) * protect
    # Only displace where the cat mask is on; elsewhere identity.
    src_x = src_x * mask + (xx + xa) * (1.0 - mask)
    src_y = src_y * mask + (yy + ya) * (1.0 - mask)
    # remap() output is ebox-sized (the size of the maps).
    warped = cv2.remap(out.astype(np.float32),
                       src_x.astype(np.float32), src_y.astype(np.float32),
                       interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    # Stamp original eye globes at the rigid-lift location so cubic remap and
    # any leftover field leak cannot smear the pupil, iris, catchlight or outline.
    if float(globe.max()) > 1e-4:
        mapx = (xx + xa).astype(np.float32)
        mapy = ((yy + ya) + rigid_lift).astype(np.float32)
        stamped = cv2.remap(out.astype(np.float32), mapx, mapy,
                            interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        g3 = globe[..., None]
        warped = warped * (1.0 - g3) + stamped * g3
    # Clean lower-lid rise (1–2 px): sample a little below in the already
    # head-lifted plate, never inside the globe, so the iris stays put.
    if float(lid.max()) > 1e-4 and sh.get('squint_lo', 0) > 0:
        amt = q * float(sh['squint_lo']) * lid
        lx = xx.astype(np.float32)
        ly = (yy + amt).astype(np.float32)
        lid_src = cv2.remap(warped, lx, ly,
                            interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        wlid = (lid * mask)[..., None]
        warped = warped * (1.0 - wlid) + lid_src * wlid
    blend = mask[..., None]
    src_roi = out[ya:ya + h, xa:xa + w].astype(np.float32)
    comp = src_roi * (1.0 - blend) + warped * blend
    out_w = out.copy()
    out_w[ya:ya + h, xa:xa + w] = np.uint8(np.clip(np.round(comp), 0, 255))
    # Lift at the mouth centre, used to paint the crisp mouth after the head moves.
    x0, y0, _, _ = BOX[n]
    my = np.clip((y0 + 20) - ya, 0, h - 1)
    fy_m = my / max(h, 1)
    lift_mouth = float(q * (
        (1.0 - sstep((fy_m - 0.58) / 0.16)) * sh['head']
        + (sstep((fy_m - 0.58) / 0.16) * (1.0 - sstep((fy_m - 0.86) / 0.12))) * sh['chest']
        + sh['tilt'] * np.exp(-0.5 * ((fy_m - 0.54) / 0.10) ** 2)
    ))
    return out_w, ebox, lift_mouth



def frame(n, t, samples=(-0.25, 0, 0.25), weights=(0.25, 0.5, 0.25)):
    global T_NOW
    T_NOW = t
    x0, y0, _, _ = BOX[n]
    out = v30[n].copy()
    if STYLE in ('mrrrp', 'growl'):
        p0, cg_g, _ = growl_curve(t)
        pg, cg = p0, cg_g
        body_p = p0
    else:
        p0, _, _ = curve(t)
        pg, cg, _ = p_geom(t)
        body_p = pg if pg > p0 else p0

    def _close_v30_mouth(dst, y_shift=0.0):
        """Paste the closed orig-cycle mouth (same pose) with a soft oval blend."""
        src = orig[n]
        dxx = xx - 48
        dyy = yy - 20
        oval = np.sqrt((dxx / 24) ** 2 + ((dyy - 6) / 15) ** 2)
        om = 1 - sstep((oval - 0.85) / 0.45)
        crop = dst[y0:y0 + PH, x0:x0 + PW].astype(np.float32)
        if abs(y_shift) < 1e-4:
            clsd = src[y0:y0 + PH, x0:x0 + PW].astype(np.float32)
        else:
            mx = (x0 + xx).astype(np.float32)
            my = (y0 + y_shift + yy).astype(np.float32)
            clsd = cv2.remap(src.astype(np.float32), mx, my,
                             interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        dst = dst.copy()
        dst[y0:y0 + PH, x0:x0 + PW] = np.uint8(np.clip(np.round(
            crop * (1 - om[..., None]) + clsd * om[..., None]), 0, 255))
        return dst

    def _stamp_rest_lips(dst, lift=0.0):
        """Re-seal the lip line to orig's resting crease, rigidly translated
        with the head lift, so a jaw/pad warp cannot open a dark slit."""
        dxx = xx - 48
        dyy = yy - 20
        band = np.exp(-0.5 * (dyy / 7.0) ** 2) * (1.0 - sstep((np.abs(dxx) - 22.0) / 8.0))
        band = cv2.GaussianBlur(band.astype(np.float32), (0, 0), 1.2)
        mx = (x0 + xx).astype(np.float32)
        my = (y0 + lift + yy).astype(np.float32)
        clsd = cv2.remap(orig[n].astype(np.float32), mx, my,
                         interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        crop = dst[y0:y0 + PH, x0:x0 + PW].astype(np.float32)
        dst = dst.copy()
        w = band[..., None]
        dst[y0:y0 + PH, x0:x0 + PW] = np.uint8(np.clip(np.round(
            crop * (1.0 - w) + clsd * w), 0, 255))
        return dst

    # Sealed closed-mouth growl (15.4s): close the v30 hole, body language,
    # then re-stamp orig's resting lip line so the lips cannot show a slit.
    if STYLE == 'growl':
        out = _close_v30_mouth(out)
        if body_p <= 1e-4:
            return out, [0.0, 0.0, 0.0], _ebox(n)
        out, ebox, lift_m = apply_body(out, n, body_p)
        out = _stamp_rest_lips(out, lift=-float(lift_m))
        return out, [p0, p0, p0], ebox
    if p0 <= 1e-4:
        # mrrrp ease-out: never snap to raw v30 (that pops the hole / jaw).
        if STYLE == 'mrrrp':
            out = _close_v30_mouth(out)
            if body_p > 1e-4:
                out, ebox, lift_m = apply_body(out, n, body_p)
                out = _stamp_rest_lips(out, lift=-float(lift_m))
                return out, [p0, p0, p0], ebox
            return out, [0.0, 0.0, 0.0], _ebox(n)
        return out, [0.0, 0.0, 0.0], _ebox(n)
    if STYLE == 'mrrrp':
        out = _close_v30_mouth(out)
    orig_lifted, ebox, lift_mouth = apply_body(orig[n].copy(), n, body_p if STYLE == 'mrrrp' else pg)
    out, ebox, lift_mouth = apply_body(out, n, body_p if STYLE == 'mrrrp' else pg)
    if STYLE == 'mrrrp':
        out = _stamp_rest_lips(out, lift=-float(lift_mouth))
        return out, [p0, p0, p0], ebox
    # Subpixel placement of the painted mouth (no 1 px quantisation pop).
    y_off = -float(lift_mouth)
    y1 = y0 + y_off
    mx = (x0 + xx).astype(np.float32)
    my = (y1 + yy).astype(np.float32)
    dst = cv2.remap(out.astype(np.float32), mx, my,
                    interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    dxx = xx - 48
    dyy = yy - 20
    oval = np.sqrt((dxx / 24) ** 2 + ((dyy - 6) / 15) ** 2)
    om = 1 - sstep((oval - 0.85) / 0.4)
    srcbox = out.astype(np.float32).copy()
    # orig_lifted is already body-warped in full-frame space; sample it at the
    # same subpixel mouth box as dst.
    orig_crop = cv2.remap(orig_lifted.astype(np.float32), mx, my,
                          interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    mouth_src = dst * (1 - om[..., None]) + orig_crop * om[..., None]
    # splatting mouth_src into srcbox at (x0, y1) happens via the same maps
    # when render_one remaps from srcbox using float y0+y_off. Keep a float
    # y_off on render_one (cy includes it). Also stamp mouth_src into srcbox
    # at the rounded location as a fallback for integer sampling.
    yi = int(round(y1))
    yi = max(0, min(720 - PH, yi))
    srcbox[yi:yi + PH, x0:x0 + PW] = mouth_src
    acc = np.zeros((PH, PW, 3), np.float32)
    ps = []
    for s_, wt in zip(samples, weights):
        if STYLE == 'mrrrp':
            p, cl, tg = growl_curve(t + s_ / 24)
            pg_s, cg_s = p, cl
        else:
            p, cl, tg = curve(t + s_ / 24)
            pg_s, cg_s, _ = p_geom(t + s_ / 24)
        ps.append(p)
        acc += wt * render_one(n, pg_s, cg_s, tg if tg is not None else 10.0, srcbox, y_off=y_off)
    m = EW[..., None]
    res = dst * (1 - m) + acc * m
    pad = 2
    xa = max(0, x0 - pad)
    ya = max(0, int(np.floor(y1)) - pad)
    xb = min(1280, x0 + PW + pad)
    yb = min(720, int(np.ceil(y1 + PH)) + pad)
    yyb, xxb = np.mgrid[ya:yb, xa:xb].astype(np.float32)
    srcx = (xxb - x0).astype(np.float32)
    srcy = (yyb - y1).astype(np.float32)
    splatted = cv2.remap(res, srcx, srcy,
                         interpolation=cv2.INTER_CUBIC,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    inx = sstep(srcx / 1.0) * sstep((PW - 1 - srcx) / 1.0)
    iny = sstep(srcy / 1.0) * sstep((PH - 1 - srcy) / 1.0)
    cov = inx * iny * cv2.remap(EW, srcx, srcy,
                                interpolation=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    cov = np.clip(cov, 0, 1)
    roi = out[ya:yb, xa:xb].astype(np.float32)
    out[ya:yb, xa:xb] = np.uint8(np.clip(np.round(
        roi * (1.0 - cov[..., None]) + splatted * cov[..., None]), 0, 255))
    return out, ps, ebox


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--preview', action='store_true')
    ap.add_argument('--style', default='all', choices=['all','medium','big','mrrrp','growl'])
    args = ap.parse_args()
    styles = ['medium','mrrrp','growl','big'] if args.style == 'all' else [args.style]
    preview = args.preview
    all_logs = {}
    for style in styles:
        STYLE = style
        frames = [f.copy() for f in v30]
        log = []
        targets = [34, 35, 36, 38, 44, 50, 54, 58, 90, 96, 100, 101, 108, 109] if preview else list(active)
        print(f'=== style {style} ===', flush=True)
        for n in targets:
            if n not in set(active):
                continue
            t = n / 24
            packed = frame(n, t)
            f, ps = packed[0], packed[1]
            ebox = packed[2] if len(packed) > 2 else _ebox(n)
            frames[n] = f
            p, cl, tg = curve(t)
            sh = shape()
            pg = p_geom(t)[0]
            rec = {
                'frame': n, 'phase_s': t, 'p_center_sample': p, 'p_subsamples': ps,
                'closing': cl, 'gs_time_s': tg, 'box': BOX[n],
                'expanded_box': ebox, 'style': style,
                'center_opening_px': float(SV * A * (sh['LIFT_U'] + sh['DROP_U']) * pg),
                'p_geom': float(pg),
                'upper_lip_lift_px': float(SV * A * sh['LIFT_U'] * pg),
                'jaw_drop_px': float(SV * A * sh['DROP_U'] * pg),
            }
            log.append(rec)
            print(f'  frame {n} p={p:.3f} p_geom={pg:.3f} '
                  f'open={rec["center_opening_px"]:.2f}px '
                  f'lift={rec["upper_lip_lift_px"]:.2f} drop={rec["jaw_drop_px"]:.2f}', flush=True)
            if preview:
                x0,y0,w,h = BOX[n]
                y0p=max(0,y0-8); hp=h+8
                crop = f[y0p:y0p+hp, x0:x0+w]
                crop4 = cv2.resize(crop, (w*4, hp*4), interpolation=cv2.INTER_NEAREST)
                mid = int(round(MOUTH_MID * 4))
                cv2.line(crop4, (mid, 0), (mid, crop4.shape[0]-1), (0, 220, 255), 1)
                cv2.imwrite(str(P/'qc'/f'preview-v41-{style}-f{n:03d}-4x.png'), crop4)
                ex,ey,ew,eh = ebox
                head = f[ey:ey+eh, ex:ex+ew]
                head2 = cv2.resize(head, (ew*2, eh*2), interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(P/'qc'/f'preview-v41-{style}-head-f{n:03d}-2x.png'), head2)
                print(f'    ebox={ebox}', flush=True)
        all_logs[style] = log
        if preview:
            continue
        dest = P / 'cycle' / f'v41-{style}-cycle-lossless.mkv'
        dest.parent.mkdir(exist_ok=True)
        enc = sp.Popen([
            'ffmpeg', '-v', 'error', '-y',
            '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', '1280x720', '-r', '24', '-i', '-',
            '-an', '-c:v', 'ffv1', '-level', '3', '-pix_fmt', 'bgr0', str(dest)
        ], stdin=sp.PIPE)
        for f in frames:
            enc.stdin.write(f.tobytes())
        enc.stdin.close()
        assert enc.wait() == 0
        np.save(P / 'cycle' / f'v41-{style}-cycle.npy', np.stack(frames))
        json.dump({
            'style': style, 'A': A, 'SV': SV, 'MEOWS': MEOWS, 'LEAD': LEAD, 'ATT': ATT,
            'SCALE': SCALE, 'active_frames': active, 'records': log,
            'shape': SHAPES[style],
            'note': 'v41 style cache; rigid eyes + clean lower-lid squint; non-active frames are v30'
        }, open(P / 'cycle' / f'v41-{style}-params.json', 'w'), indent=1)
        for n in range(N):
            if n not in active:
                assert np.array_equal(frames[n], v30[n]), (style, n)
            else:
                rec = next((r for r in log if r['frame']==n), None)
                if rec is None:
                    continue
                x0, y0, w, h = rec['expanded_box']
                m = np.ones((720, 1280), bool)
                m[y0:y0 + h, x0:x0 + w] = False
                assert np.array_equal(frames[n][m], v30[n][m]), (style, n)
        print('ok', style, [(r['frame'], round(r['p_center_sample'], 3)) for r in log])
    if preview:
        print('preview only')
        raise SystemExit(0)
    json.dump({'styles': list(all_logs), 'active_frames': active},
              open(P / 'cycle' / 'v41-index.json', 'w'), indent=1)
    print('all styles written')
