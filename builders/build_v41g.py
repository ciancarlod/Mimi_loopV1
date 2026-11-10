#!/usr/bin/env python3
"""v41g: tail idle, right-corner plant brush, and dust motes on frozen v41f-r2.

Does not overwrite v41 / v41b / v41c / v41d / v41e / v41e-r2 / v41f / v41f-r2.

Source is the v41f-r2 lossless grade (atmosphere already locked). This
pass only adds recipes 5–7 from LOFI-RESEARCH.md, all 241-periodic.

Honest anatomy note: the v24/v30 desk crop has no hanging S-curve tail.
The visible mass that almost touches the right-corner plant is the
lower-right haunch / flank silhouette (tip ≈ x1056, y624). That edge
is what we idle and curl; we do not paint a fake hanging tail.
"""
from __future__ import annotations
import hashlib, subprocess as sp, sys
from pathlib import Path
import cv2
import numpy as np

P = Path('/workspace/lofi-v41-meow')
QC = P / 'qc' / 'v41g'
QC.mkdir(parents=True, exist_ok=True)
(QC / 'preview').mkdir(exist_ok=True)
(QC / 'encoded').mkdir(exist_ok=True)
WORK = P / 'v41g'
WORK.mkdir(exist_ok=True)

SRC = P / 'v41f' / 'v41f-r2-span-890-915-lossless.mkv'
PCM = P / 'work' / 'test-audio-890-915-v41.f32le'
MATTE = P / 'v41f' / 'cat-alpha-890-915.mkv'
V41_MP4 = P / 'v41-test-25s.mp4'
V41D_MP4 = P / 'v41d-test-25s.mp4'
V41E_MP4 = P / 'v41e-test-25s.mp4'
V41ER2_MP4 = P / 'v41e-r2-test-25s.mp4'
V41F_MP4 = P / 'v41f-test-25s.mp4'
V41FR2_MP4 = P / 'v41f-r2-test-25s.mp4'
V41_SHA = 'f53cb1127c75775b740b555dd856e0002c784f80a2a3097118d8e197a794e95f'
V41D_SHA = 'd1dab25187f4756969344036f50cd0ef89fac7c414d80006b8853acf04be2b34'
V41E_SHA = '211f5bba5d04ae55234ebedb5d7b9e2abedcdccdcc54e6cce827fd724'
# note: leave exact sha below; typo guard in the real constant
V41E_SHA = '211f5bba5d04ae55234ebedb5d7b1ce1b7b9e2abedcdccdcc54e6cce827fd724'
V41ER2_SHA = 'bc19246dc513443cb78cc15a9ff4054be5fa66a73b59740a8c5c31bae74986ff'
V41F_SHA = 'b697e8b6e223e94a0e26a97abeb85b29764c70a35037680eed6b936e138ca703'
V41FR2_SHA = 'e7db48cc7d147a9e8ed86568c4583dc7bf1559911ff98b846194b2795171ab50'
V41FR2_LOS_SHA = '3239b6e01c251bc936390b8c09cbdae0d64f484a2f20c40eecfb37b63efbb299'
PCM_SHA = 'a635074f164c14fb3faabbf33c258703e9b7b9645eddbf8028f63a29573745a0'
NFR, W, H, FPS, NCYCLE = 600, 1280, 720, 24, 241

sys.path.insert(0, str(P))
import build_v41c as C

LX, LY = 323.0, 148.0
# Haunch "bone": base just below the mic / mid-flank, tip at the
# silhouette point closest to the right-corner plant.
BASE = np.array([980.0, 518.0], np.float32)
TIP = np.array([1056.0, 628.0], np.float32)
# Stem / pot lock for the right-corner plant (bottom-right, mostly off-frame).
STEM = np.array([1228.0, 716.0], np.float32)
# Nearest leaf contact (just right of the haunch bulge).
CONTACT = np.array([1068.0, 640.0], np.float32)

TITLE_Y0, TITLE_Y1, TITLE_X0, TITLE_X1 = 14, 60, 16, 165


def sstep(x):
    x = np.clip(x, 0, 1)
    return x * x * x * (x * (6 * x - 15) + 10)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for ch in iter(lambda: f.read(1 << 20), b''):
            h.update(ch)
    return h.hexdigest()


def phase_of(i):
    return (152 + i) % NCYCLE


def luma(im):
    im = im.astype(np.float32)
    return 0.114 * im[..., 0] + 0.587 * im[..., 1] + 0.299 * im[..., 2]


def load_span(path, gray=False):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if gray:
            if f.ndim == 3:
                f = f[..., 0] if f.shape[2] == 1 else cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        frames.append(f)
    cap.release()
    return frames


Y, X = np.mgrid[:H, :W].astype(np.float32)


def encode_mp4(frames, path):
    enc = sp.Popen([
        'ffmpeg', '-v', 'error', '-y',
        '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
        '-f', 'f32le', '-ar', '48000', '-ac', '2', '-i', str(PCM),
        '-c:v', 'libx264', '-preset', 'slow', '-crf', '12', '-pix_fmt', 'yuv420p',
        '-c:a', 'aac', '-b:a', '320k', '-ar', '48000', '-movflags', '+faststart', '-shortest',
        str(path)
    ], stdin=sp.PIPE)
    for f in frames:
        enc.stdin.write(f.tobytes())
    enc.stdin.close()
    assert enc.wait() == 0, 'encode failed'


def write_lossless(frames, path):
    enc = sp.Popen([
        'ffmpeg', '-v', 'error', '-y',
        '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
        '-an', '-c:v', 'ffv1', '-level', '3', '-pix_fmt', 'bgr0', str(path)
    ], stdin=sp.PIPE)
    for f in frames:
        enc.stdin.write(f.tobytes())
    enc.stdin.close()
    assert enc.wait() == 0


def face_protect():
    """Soft mask of face / mouth / eyes / upper head — dust and warps stay off it."""
    mouth = C.mouth_mask()
    eyes = C.eye_mask()
    # Generous head / cheek / ear coverage — mote kernels (~8 px) must
    # not spill a warm highlight onto the fur even if the centre is off-face.
    head = sstep(1.0 - np.sqrt(((X - 690.0) / 240.0) ** 2 + ((Y - 360.0) / 210.0) ** 2))
    return np.clip(np.maximum.reduce([mouth, eyes, head, C.title_mask()]), 0, 1)


def pane_solid():
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [
        np.array([[266, 90], [560, 90], [558, 394], [268, 396]], np.int32),
        np.array([[600, 90], [828, 92], [826, 396], [600, 396]], np.int32),
    ], 255)
    return cv2.GaussianBlur(m.astype(np.float32) / 255.0, (0, 0), 2.0)


# ---------- plant mask (static plate; cat subtracted per frame) ----------
def build_plant_plate(fr, cat0):
    b, g, r = [c.astype(np.float32) for c in cv2.split(fr)]
    roi = np.zeros((H, W), np.float32)
    roi[350:H, 1020:W] = 1.0
    cat_d = cv2.dilate((cat0 > 0.35).astype(np.uint8),
                       cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    chair = ((r > g + 10) & (r > b + 8) & (r > 25)).astype(np.float32)
    plant = roi * (1.0 - cat_d) * (1.0 - chair)
    # keep foliage: relatively green, or dark leaf in the plant box
    leafy = ((g + 6 >= r) & (g >= b - 10) & (g > 8)).astype(np.float32)
    plant = plant * leafy
    plant = (plant > 0.5).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    plant = cv2.morphologyEx(plant, cv2.MORPH_CLOSE, k)
    plant = cv2.morphologyEx(plant, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    # The right-corner floor plant only — drop hanging ivy / shelf foliage
    # (those sit higher, y≲460, and must stay still).
    plant[:460, :] = 0
    n, lab, stats, _ = cv2.connectedComponentsWithStats(plant, 8)
    keep = np.zeros_like(plant)
    for i in range(1, n):
        y = stats[i, cv2.CC_STAT_TOP]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        # keep blobs that reach the lower plant (intersect y>520)
        if stats[i, cv2.CC_STAT_AREA] >= 80 and (y + h) > 520:
            keep[lab == i] = 1
    # lock pot / stem base
    dist_stem = np.hypot(X - STEM[0], Y - STEM[1])
    pot = sstep((48.0 - dist_stem) / 22.0) * sstep((Y - 680.0) / 20.0)
    plate = cv2.GaussianBlur(keep.astype(np.float32), (0, 0), 1.1)
    plate = np.clip(plate * (1.0 - 0.95 * pot), 0, 1)
    return plate


# ---------- recipe 6: haunch / tail-edge idle ----------
BONE = TIP - BASE
BLEN = float(np.hypot(*BONE))
BONE_U = BONE / BLEN
# Image-space +rotation (y down) moves the tip toward the plant: right and slightly up.
PERP = np.array([BONE_U[1], -BONE_U[0]], np.float32)  # (uy, -ux wait): u=(ux,uy), perp=(uy, -ux)
PERP = np.array([BONE_U[1], -BONE_U[0]], np.float32)


def tail_theta(ph):
    """Radians at the tip. 2-cycle pendulum + 5-cycle twitch + one curl per loop.

    The 2-cycle term is phased so its plant-ward peak lands at ph 110,
    coinciding with the recipe-6 curl and the recipe-7 brush.
    """
    A1 = 11.0 / BLEN          # ~11 px at the tip
    A2 = 0.35 * A1
    # 2π*2*110/241 + psi2 = π/2  →  tip at +A1 on the plant-ward side
    psi2 = np.pi / 2.0 - 2 * np.pi * 2 * 110.0 / NCYCLE
    phi = 0.40
    th = A1 * np.sin(2 * np.pi * 2 * ph / NCYCLE + psi2)
    th += A2 * np.sin(2 * np.pi * 5 * ph / NCYCLE + phi)
    # raised-cosine curl, 0 outside [90, 130], +5 px at mid (ph 110)
    if 90.0 < ph < 130.0:
        u = (ph - 90.0) / 40.0
        bump = 0.5 - 0.5 * np.cos(2 * np.pi * u)
        th += (5.0 / BLEN) * bump
    return float(th)


def tail_weights(alpha):
    """Spatial weight along the haunch bone. 0 on hips / back / head / mic."""
    vx = X - BASE[0]
    vy = Y - BASE[1]
    t = (vx * BONE_U[0] + vy * BONE_U[1]) / BLEN          # 0 at base, 1 at tip
    rad = np.abs(vx * PERP[0] + vy * PERP[1])
    # allow a little past the tip so remap can land the edge onto the plant
    along = sstep((t - 0.02) / 0.28) * sstep((1.18 - t) / 0.18)
    # narrower near the base (don't drag the torso), wider near the tip
    half = 28.0 + 38.0 * np.clip(t, 0, 1)
    across = sstep((half - rad) / 14.0)
    # hard spatial gate: below the mic, right of the chest
    gate = sstep((Y - 470.0) / 22.0) * sstep((X - 930.0) / 20.0)
    w = along * across * gate * np.clip(alpha, 0, 1)
    w = cv2.GaussianBlur(w, (0, 0), 1.6)
    return np.clip(w, 0, 1)


def apply_tail(img, alpha, ph):
    th = tail_theta(ph)
    w = tail_weights(alpha)
    # rotation around BASE, scaled by along-bone parameter (tip leads)
    vx = X - BASE[0]
    vy = Y - BASE[1]
    t = np.clip((vx * BONE_U[0] + vy * BONE_U[1]) / BLEN, 0, 1.15)
    lead = sstep(t / 0.95)          # 0 at base, 1 at tip
    ang = th * lead
    ca, sa = np.cos(ang), np.sin(ang)
    # dest = BASE + R(ang) * (p-BASE); source = inverse rot of dest
    # For remap, map = R(-ang) applied at dest ≈ p - ang * perp_vec (small angle)
    # Exact: src = BASE + R(-ang) * (p - BASE)
    sx = BASE[0] + ca * vx + sa * vy
    sy = BASE[1] - sa * vx + ca * vy
    # Only evaluate remap in a padded ROI
    y0, y1, x0, x1 = 450, H, 860, W
    mapx = X.copy()
    mapy = Y.copy()
    mapx[y0:y1, x0:x1] = sx[y0:y1, x0:x1]
    mapy[y0:y1, x0:x1] = sy[y0:y1, x0:x1]
    warped = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    wa = cv2.remap(w, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    wa = np.clip(np.maximum(w, wa), 0, 1)  # cover both vacated stretch and new edge
    # Don't let the warp bleed onto locked plant pot / far uninvolved chair:
    # wa is already cat-weighted.
    out = img.astype(np.float32) * (1.0 - wa[..., None]) + warped.astype(np.float32) * wa[..., None]
    # Very thin rim: warm on the lamp-facing (left/upper) moving edge,
    # cooler and weaker toward the plant / window.
    if wa.max() > 0.05:
        edge = cv2.Sobel(wa, cv2.CV_32F, 1, 0, ksize=3)
        # positive Sobel-x = right-facing edge (plant side)
        right_e = np.clip(edge / 2.5, 0, 1) * wa
        left_e = np.clip(-edge / 2.5, 0, 1) * wa
        right_e = cv2.GaussianBlur(right_e, (0, 0), 0.8)
        left_e = cv2.GaussianBlur(left_e, (0, 0), 0.8)
        warm = np.array([0xC0, 0xE6, 0xFF], np.float32)  # BGR of #FFE6C0
        cool = np.array([0xC8, 0xB0, 0x96], np.float32)
        out = out + left_e[..., None] * (warm - out) * 0.16
        out = out + right_e[..., None] * (cool - out) * 0.08
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------- recipe 7: right-corner plant brush ----------
IMPULSE_A, IMPULSE_B = 100.0, 112.0
SPRING_T = 1.80    # seconds
SPRING_TAU = 1.55  # seconds; <1 px by frame 241
PUSH_PX = 10.0
# Push direction: away from the cat, into the plant (right, slightly down).
PUSH = np.array([0.92, 0.39], np.float32)
PUSH = PUSH / np.hypot(*PUSH)


def plant_amp(ph):
    """Scalar 0..PUSH_PX, 241-periodic, ~0 at ph 0 / 241."""
    if IMPULSE_A <= ph <= IMPULSE_B:
        u = (ph - IMPULSE_A) / (IMPULSE_B - IMPULSE_A)
        return PUSH_PX * sstep(u)
    if ph > IMPULSE_B:
        tsec = (ph - IMPULSE_B) / float(FPS)
        raw = PUSH_PX * np.exp(-tsec / SPRING_TAU) * np.cos(2 * np.pi * tsec / SPRING_T)
        # fade the last ~9 frames so ph 241 / 0 is a clean zero (sub-px anyway)
        fade = 1.0 if ph < 232.0 else sstep((241.0 - ph) / 9.0)
        return float(raw * fade)
    return 0.0


def plant_field(plate, cat_a):
    """Per-pixel amplitude 0..1 and a soft live mask (plate minus this frame's cat)."""
    cat_d = cv2.GaussianBlur(
        cv2.dilate((cat_a > 0.35).astype(np.uint8),
                   cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))).astype(np.float32),
        (0, 0), 1.2)
    live = np.clip(plate * (1.0 - cat_d), 0, 1)
    d_contact = np.hypot(X - CONTACT[0], Y - CONTACT[1])
    # nearest leaves ~1, farther leaves 0.35–0.5, pot already zeroed on the plate
    near = 0.38 + 0.62 * sstep((260.0 - d_contact) / 200.0)
    # rotation around the stem: farther from stem = more travel
    d_stem = np.hypot(X - STEM[0], Y - STEM[1])
    span = sstep((d_stem - 36.0) / 80.0)
    return live, np.clip(near * span, 0, 1)


def apply_plant(img, plate, cat_a, ph):
    A = plant_amp(ph)
    if abs(A) < 0.04:
        return img
    live, amp = plant_field(plate, cat_a)
    # Mostly a small rotation about the stem (3.8° peak on the nearest
    # leaves ≈ 10–12 px) plus a short 4 px push. Larger combined travel
    # left holes and a doubled leaf tip in r1 of this preview.
    ang = np.radians(3.8) * (A / PUSH_PX) * amp
    trans = 4.0 * (A / PUSH_PX) * amp
    vx = X - STEM[0]
    vy = Y - STEM[1]
    ca, sa = np.cos(ang), np.sin(ang)
    sx = STEM[0] + ca * vx + sa * vy - (trans * PUSH[0])
    sy = STEM[1] - sa * vx + ca * vy - (trans * PUSH[1])
    y0, y1, x0, x1 = 450, H, 1000, W
    mapx = X.copy()
    mapy = Y.copy()
    mapx[y0:y1, x0:x1] = sx[y0:y1, x0:x1]
    mapy[y0:y1, x0:x1] = sy[y0:y1, x0:x1]
    warped = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    wa = cv2.remap(live, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    wa = np.clip(wa, 0, 1)
    # Keep the cat locked — dilate so a moving leaf cannot smear into fur.
    cat_d = cv2.GaussianBlur(
        cv2.dilate((cat_a > 0.30).astype(np.uint8),
                   cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))).astype(np.float32),
        (0, 0), 1.4)
    wa = wa * (1.0 - np.clip(cat_d, 0, 1))
    # Vacated leaf pixels (old position, not covered by the warped leaf)
    # are inpainted from the chair / neighbouring foliage — do not leave
    # a stretched ghost of the same leaf by also keeping `live` in the blend.
    vacated = np.clip(live * (1.0 - wa) * (1.0 - cat_d), 0, 1)
    vmask = (vacated > 0.28).astype(np.uint8) * 255
    if vmask.any():
        filled = cv2.inpaint(img, vmask, 4, cv2.INPAINT_TELEA)
        vf = cv2.GaussianBlur(vacated, (0, 0), 1.0)
        base = img.astype(np.float32) * (1.0 - vf[..., None]) + filled.astype(np.float32) * vf[..., None]
    else:
        base = img.astype(np.float32)
    out = base * (1.0 - wa[..., None]) + warped.astype(np.float32) * wa[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------- recipe 5: dust motes ----------
def _make_motes():
    rng = np.random.RandomState(41)
    motes = []
    # 28 in the lamp cone
    for i in range(28):
        # sample along the cone axis
        along = rng.uniform(40, 420)
        axis = np.array([0.20, 1.00], np.float32)
        axis = axis / np.hypot(*axis)
        hw = 16.0 + along * 0.48
        perp = rng.uniform(-0.85, 0.85) * hw
        # perp vector
        pr = np.array([axis[1], -axis[0]])
        p = np.array([LX, LY]) + along * axis + perp * pr
        layer = int(rng.choice([0, 1, 1, 2]))  # mid more common
        k = int(rng.choice([1, 1, 2, 2, 3]))
        motes.append(dict(
            x0=float(np.clip(p[0], 80, 820)),
            y0=float(np.clip(p[1], 90, 580)),
            rx=rng.uniform(8, 22),
            ry=rng.uniform(10, 28),
            jx=rng.uniform(0.3, 0.8),
            jy=rng.uniform(0.3, 0.8),
            k=k,
            p1=rng.uniform(0, 2 * np.pi),
            p2=rng.uniform(0, 2 * np.pi),
            p3=rng.uniform(0, 2 * np.pi),
            layer=layer,
            cone=True,
            peak=rng.uniform(0.60, 0.90),
        ))
    # 8 dimmer mid-air (room, not glass)
    for i in range(8):
        motes.append(dict(
            x0=float(rng.uniform(160, 980)),
            y0=float(rng.uniform(80, 500)),
            rx=rng.uniform(6, 16),
            ry=rng.uniform(8, 18),
            jx=rng.uniform(0.3, 0.6),
            jy=rng.uniform(0.3, 0.6),
            k=int(rng.choice([1, 2])),
            p1=rng.uniform(0, 2 * np.pi),
            p2=rng.uniform(0, 2 * np.pi),
            p3=rng.uniform(0, 2 * np.pi),
            layer=int(rng.choice([1, 2, 2])),
            cone=False,
            peak=rng.uniform(0.08, 0.15),
        ))
    return motes


MOTES = _make_motes()
# layer: 0 near (larger, slight blur), 1 mid (sharp), 2 far (small, softer)
LAYER = {
    0: dict(core=3.2, sig=1.0, halo=3.2),
    1: dict(core=2.0, sig=0.4, halo=0.0),
    2: dict(core=1.15, sig=1.5, halo=0.0),
}


def _kernel(core, sig, halo, color, peak):
    rad = int(np.ceil(max(core, halo) + 3 * max(sig, 0.8) + 2))
    s = 2 * rad + 1
    yy, xx = np.mgrid[-rad:rad + 1, -rad:rad + 1].astype(np.float32)
    r = np.hypot(xx, yy)
    g = np.exp(-0.5 * (r / max(core * 0.45, 0.4)) ** 2)
    if sig > 0.2:
        g = cv2.GaussianBlur(g, (0, 0), sig)
    if halo > 0:
        h = 0.28 * np.exp(-0.5 * (r / halo) ** 2)
        g = np.maximum(g, h)
    g = g / (g.max() + 1e-6) * peak
    ker = (g[..., None] * color[None, None, :]).astype(np.float32)
    return ker, rad


_KER_CACHE = {}


def mote_kernel(m):
    key = (m['layer'], m['cone'], round(m['peak'], 3))
    if key not in _KER_CACHE:
        L = LAYER[m['layer']]
        if m['cone']:
            col = np.array([0xC0, 0xE6, 0xFF], np.float32)  # #FFE6C0
            peak = m['peak']
            halo = L['halo'] if m['layer'] == 0 else 0.0
        else:
            col = np.array([0xA8, 0xC8, 0xD8], np.float32)  # #D8C8A8
            peak = m['peak']
            halo = 0.0
        _KER_CACHE[key] = _kernel(L['core'], L['sig'], halo, col, peak)
    return _KER_CACHE[key]


def cone_w(x, y):
    axis = np.array([0.20, 1.00], np.float32)
    axis = axis / np.hypot(*axis)
    vx, vy = x - LX, y - LY
    along = vx * axis[0] + vy * axis[1]
    perp = abs(vx * axis[1] - vy * axis[0])
    hw = 16.0 + np.maximum(along, 0) * 0.48
    return float(sstep((along - 6) / 16.0) * sstep((hw - perp) / 12.0))


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
        if xi < 8 or xi >= W - 8 or yi < 8 or yi >= H - 8:
            continue
        # skip face / title / mouth / window glass (don't fight snow)
        if face[yi, xi] > 0.25:
            continue
        if panes[yi, xi] > 0.25:
            continue
        # stay off fur (and a dilated halo so the kernel cannot kiss a cheek)
        if yi < 600 and alpha[max(0, yi - 8):yi + 9, max(0, xi - 8):xi + 9].max() > 0.35:
            continue
        ker, rad = mote_kernel(m)
        # dim if this cone mote has drifted out of the beam
        scale = 1.0
        if m['cone']:
            cw = cone_w(x, y)
            scale = 0.12 + 0.88 * cw
        y0, y1 = yi - rad, yi + rad + 1
        x0, x1 = xi - rad, xi + rad + 1
        if y0 < 0 or x0 < 0 or y1 > H or x1 > W:
            continue
        dust[y0:y1, x0:x1] += ker * scale
    # master opacity 0.65; screen
    op = 0.65
    base = img.astype(np.float32)
    fur = cv2.dilate((alpha > 0.35).astype(np.uint8),
                     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))).astype(np.float32)
    fur = cv2.GaussianBlur(fur, (0, 0), 2.0) * sstep((620.0 - Y) / 24.0)
    prot = np.clip(np.maximum(face, fur), 0, 1)
    dust = dust * (1.0 - prot[..., None])
    add = np.clip(dust * op, 0, 255)
    # screen: 1-(1-a)(1-b)
    out = 255.0 - (255.0 - base) * (1.0 - add / 255.0)
    # re-kill anything that landed on the face / title / fur after kernel splat
    out = base * prot[..., None] + out * (1.0 - prot[..., None])
    return np.clip(out, 0, 255).astype(np.uint8)


def process(src, mattes, plant_plate, face, panes, preview=None):
    want = set(preview) if preview else set(range(NFR))
    out = [None] * NFR
    for i in range(NFR):
        if i not in want:
            continue
        ph = phase_of(i)
        a = mattes[i].astype(np.float32)
        if a.max() > 1.5:
            a = a / 255.0
        img = src[i]
        img = apply_tail(img, a, ph)
        img = apply_plant(img, plant_plate, a, ph)
        img = apply_dust(img, a, ph, face, panes)
        img = img.copy()
        img[TITLE_Y0:TITLE_Y1, TITLE_X0:TITLE_X1] = src[i][TITLE_Y0:TITLE_Y1, TITLE_X0:TITLE_X1]
        out[i] = img
        if i % 50 == 0 or preview:
            print(f'  n{i} ph{ph} tail_th={tail_theta(ph)*BLEN:.2f}px plant={plant_amp(ph):.2f}px',
                  flush=True)
    return out


def contact_sheet(frames, path, labels):
    # 4xN grid of 640x360 thumbs
    thumbs = []
    for f, lab in zip(frames, labels):
        t = cv2.resize(f, (640, 360), interpolation=cv2.INTER_AREA)
        cv2.putText(t, lab, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2, cv2.LINE_AA)
        thumbs.append(t)
    cols = 4
    rows = int(np.ceil(len(thumbs) / cols))
    while len(thumbs) < rows * cols:
        thumbs.append(np.zeros((360, 640, 3), np.uint8))
    grid = []
    for r in range(rows):
        grid.append(np.hstack(thumbs[r * cols:(r + 1) * cols]))
    sheet = np.vstack(grid)
    cv2.imwrite(str(path), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 92])


def save_preview(src, out, idx):
    prev = QC / 'preview'
    prev.mkdir(exist_ok=True)
    crops = {
        'tail': (500, 720, 900, 1160),
        'plant': (430, 720, 1000, 1280),
        'contact': (540, 720, 980, 1180),
        'lamp': (40, 280, 200, 500),
        'face': (200, 520, 520, 900),
        'mouth': (340, 500, 600, 820),
        'desk': (300, 620, 200, 700),
        'window': (70, 420, 250, 850),
    }

    def crop(im, box, scale=2):
        y0, y1, x0, x1 = box
        c = im[y0:y1, x0:x1]
        if scale != 1:
            c = cv2.resize(c, (c.shape[1] * scale, c.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
        return c

    frames_for_sheet = []
    labels = []
    for i in idx:
        ph = phase_of(i)
        p = prev / f'full-n{i}.png'
        cv2.imwrite(str(p), out[i])
        for tag, box in crops.items():
            cv2.imwrite(str(prev / f'{tag}-n{i}.jpg'), crop(out[i], box, 2),
                        [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        # 3x contact / tail edge
        cv2.imwrite(str(prev / f'contact-3x-n{i}.jpg'), crop(out[i], (560, 720, 1000, 1160), 3),
                    [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        frames_for_sheet.append(out[i])
        labels.append(f'n{i} ph{ph}')
        # diff vs source in the tail/plant box
        d = cv2.absdiff(out[i], src[i])
        roi = d[500:720, 900:1280]
        print(f'  preview n{i}: tail/plant meanabs={roi.mean():.3f} max={int(roi.max())}  '
              f'dust-desk meanabs={cv2.absdiff(out[i], src[i])[300:560, 200:700].mean():.3f}')

    contact_sheet(frames_for_sheet, prev / 'contact-sheet.jpg', labels)
    # seam n0 vs n241 (same phase 152)
    if 0 in idx and 241 in idx:
        a, b = out[0], out[241]
        d = cv2.absdiff(a, b).astype(np.float32)
        print(f'  n0 vs n241 global meanabs={d.mean():.4f} max={int(d.max())}')
        print(f'    tail box meanabs={d[500:720, 900:1160].mean():.4f}')
        print(f'    plant box meanabs={d[430:720, 1020:1280].mean():.4f}')
        print(f'    desk/dust meanabs={d[200:520, 180:700].mean():.4f}')
        # side-by-side + diffx8
        pair = np.hstack([a[500:720, 900:1280], b[500:720, 900:1280],
                          np.clip(d[500:720, 900:1280] * 8, 0, 255).astype(np.uint8)])
        cv2.imwrite(str(prev / 'seam-n0-n241-tail.jpg'), pair)
        pair2 = np.hstack([a, b, np.clip(d * 8, 0, 255).astype(np.uint8)])
        pair2 = cv2.resize(pair2, (1920, 360), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(prev / 'seam-n0-n241.jpg'), pair2)
    print('preview written', prev)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--preview', action='store_true')
    args = ap.parse_args()

    for label, path, expect in (
        ('v41', V41_MP4, V41_SHA),
        ('v41d', V41D_MP4, V41D_SHA),
        ('v41e', V41E_MP4, V41E_SHA),
        ('v41e-r2', V41ER2_MP4, V41ER2_SHA),
        ('v41f', V41F_MP4, V41F_SHA),
        ('v41f-r2', V41FR2_MP4, V41FR2_SHA),
        ('v41f-r2-lossless', SRC, V41FR2_LOS_SHA),
        ('pcm', PCM, PCM_SHA),
    ):
        got = sha256(path)
        if got != expect:
            raise SystemExit(f'{label} sha changed: {got}')
    print('frozen shas ok', flush=True)

    print('loading v41f-r2 lossless…', flush=True)
    src = load_span(SRC)
    assert len(src) == NFR
    print('loading cat mattes…', flush=True)
    mattes = load_span(MATTE, gray=True)
    assert len(mattes) == NFR
    a0 = mattes[0].astype(np.float32)
    if a0.max() > 1.5:
        a0 = a0 / 255.0
    plant_plate = build_plant_plate(src[0], a0)
    cv2.imwrite(str(QC / 'preview' / 'plant-plate.png'), (plant_plate * 255).astype(np.uint8))
    print('plant plate px', float((plant_plate > 0.3).sum()), flush=True)
    face = face_protect()
    panes = pane_solid()

    if args.preview:
        idx = [0, 60, 88, 89, 120, 138, 180, 189, 199, 219, 241, 370, 432, 440]
        out = process(src, mattes, plant_plate, face, panes, preview=idx)
        save_preview(src, out, idx)
        return

    out = process(src, mattes, plant_plate, face, panes)
    bad = [i for i, (a, b) in enumerate(zip(src, out))
           if not np.array_equal(a[TITLE_Y0:TITLE_Y1, TITLE_X0:TITLE_X1],
                                 b[TITLE_Y0:TITLE_Y1, TITLE_X0:TITLE_X1])]
    print('title mismatches' if bad else 'title core byte-identical', len(bad) if bad else '')
    los = WORK / 'v41g-span-890-915-lossless.mkv'
    mp4 = P / 'v41g-test-25s.mp4'
    print('writing lossless…', flush=True)
    write_lossless(out, los)
    print('encoding mp4…', flush=True)
    encode_mp4(out, mp4)
    print('v41g mp4', sha256(mp4))
    print('v41g lossless', sha256(los))
    print('v41 still', sha256(V41_MP4))
    print('v41d still', sha256(V41D_MP4))
    print('v41e still', sha256(V41E_MP4))
    print('v41e-r2 still', sha256(V41ER2_MP4))
    print('v41f still', sha256(V41F_MP4))
    print('v41f-r2 still', sha256(V41FR2_MP4))


if __name__ == '__main__':
    main()
