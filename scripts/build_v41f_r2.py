#!/usr/bin/env python3
"""v41f-r2: window-sky and lower-landscape fix on the frozen v41f grade.

Keeps v41 / v41b / v41c / v41d / v41e / v41e-r2 / v41f untouched.
Built from the same v41d lossless as v41f, not from v41f pixels.


Fixes the v41e/r2 disasters (rectangular grade holes on the face and
window, blue leaks into fur, a cooled-down lamp, a room that barely
darkened). Does not overwrite v41 / v41b / v41c / v41d / v41e / v41e-r2.

Window: hand-drawn pane polygons (glass only), minus a rembg cat
silhouette, minus plant leaves and the lamp. A painted dusk sky plate
is recomposed under the trees/snow via a low-frequency sky key; falling
snow is kept as the high-frequency residual.

Room: smooth radial / downward-cone falloff from the lamp. The shade,
bulb and glow are excluded from darkening and given a warm lift
(#FFF0C9 core + warm bloom + warm desk pool). The cat is darkened
gently, warmer on the lamp side.

Targets: global luma about 40–48; lamp-pool to far-corner ratio >= 3.

Tail / dust / plant motion are NOT in this pass (v41g, after this passes).
"""
from __future__ import annotations
import hashlib, subprocess as sp, sys
from pathlib import Path
import cv2
import numpy as np

P = Path('/workspace/lofi-v41-meow')
QC = P / 'qc' / 'v41f-r2'
QC.mkdir(parents=True, exist_ok=True)
(QC / 'preview').mkdir(exist_ok=True)
(QC / 'encoded').mkdir(exist_ok=True)
WORK = P / 'v41f'
WORK.mkdir(exist_ok=True)

SRC = P / 'v41d' / 'v41d-span-890-915-lossless.mkv'
PCM = P / 'work' / 'test-audio-890-915-v41.f32le'
MATTE = WORK / 'cat-alpha-890-915.mkv'
MATTE_PNG = QC / 'mattes'
V41_MP4 = P / 'v41-test-25s.mp4'
V41D_MP4 = P / 'v41d-test-25s.mp4'
V41E_MP4 = P / 'v41e-test-25s.mp4'
V41ER2_MP4 = P / 'v41e-r2-test-25s.mp4'
V41F_MP4 = P / 'v41f-test-25s.mp4'
V41_SHA = 'f53cb1127c75775b740b555dd856e0002c784f80a2a3097118d8e197a794e95f'
V41D_SHA = 'd1dab25187f4756969344036f50cd0ef89fac7c414d80006b8853acf04be2b34'
V41E_SHA = '211f5bba5d04ae55234ebedb5d7b1ce1b7b9e2abedcdccdcc54e6cce827fd724'
V41ER2_SHA = 'bc19246dc513443cb78cc15a9ff4054be5fa66a73b59740a8c5c31bae74986ff'
V41F_SHA = 'b697e8b6e223e94a0e26a97abeb85b29764c70a35037680eed6b936e138ca703'
PCM_SHA = 'a635074f164c14fb3faabbf33c258703e9b7b9645eddbf8028f63a29573745a0'
NFR, W, H, FPS, NCYCLE = 600, 1280, 720, 24, 241

sys.path.insert(0, str(P))
import build_v41c as C

# Lamp bulb (from v28 / measured on v41d n0)
LX, LY = 323.0, 148.0

# Hand-traced glass panes on v41d n0, inset from the wooden frame and
# the central mullion. Right pane continues behind the cat; the cat
# matte is subtracted per frame.
PANE_LEFT = np.array([[266, 90], [560, 90], [558, 394], [268, 396]], np.int32)
PANE_RIGHT = np.array([[600, 90], [828, 92], [826, 396], [600, 396]], np.int32)


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

def hex_bgr(h):
    h = h.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return np.array([b, g, r], np.float32)

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


def pane_mask():
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [PANE_LEFT, PANE_RIGHT], 255)
    # 2–3 px feather
    return cv2.GaussianBlur(m.astype(np.float32) / 255.0, (0, 0), 2.6)


def ellipse_fill(u, v, rx, ry, feather=0.14):
    """Filled ellipse: 1 inside, soft 0 at the rim. (sstep(1-r) alone is a blob.)"""
    r = np.sqrt((u / rx) ** 2 + (v / ry) ** 2)
    return sstep((1.0 - r) / feather)


def lamp_masks():
    """World-locked lamp: metal hood, warm interior, bulb, soft glow.
    Built from geometry + the v28 underside mask, not from a box classifier.
    """
    ang = np.deg2rad(-17.0)
    dx, dy = X - 323.0, Y - 132.0
    u = dx * np.cos(ang) + dy * np.sin(ang)
    v = -dx * np.sin(ang) + dy * np.cos(ang)
    # Hood / shade as a filled tilted ellipse (the metal bowl + its opening)
    hood = ellipse_fill(u, v, 118.0, 58.0, 0.12)
    # Interior (underside of the shade): wider on the opening, dropped slightly
    interior = ellipse_fill(u, v - 10.0, 108.0, 46.0, 0.12)
    interior = interior * sstep((Y - 105) / 6.0)  # mostly the opening, not the top cap
    # Bulb
    bulb = ellipse_fill(X - LX, Y - LY, 20.0, 18.0, 0.20)
    # v28 glow mask (already painted for the underside)
    v28 = cv2.imread('/workspace/lofi-v28/lamp-mask.png', 0)
    glow = np.zeros((H, W), np.float32)
    if v28 is not None:
        ly, lx = 80, 205
        h, w = v28.shape
        glow[ly:ly + h, lx:lx + w] = v28.astype(np.float32) / 255.0
    glow = cv2.GaussianBlur(np.clip(glow, 0, 1), (0, 0), 2.0)
    # Arm / stem near the fixture — keep it out of the cool room grade
    stem = ellipse_fill(X - 175.0, Y - 160.0, 38.0, 125.0, 0.16)
    stem = stem * sstep((260 - X) / 10.0) * sstep((Y - 60) / 10.0) * sstep((260 - Y) / 10.0)
    protect = np.clip(np.maximum.reduce([hood, interior, bulb, glow, stem * 0.8]), 0, 1)
    protect = cv2.GaussianBlur(protect, (0, 0), 2.0)
    # Fixture for window exclusion: metal + cream interior + bulb, no glow halo
    # (a glow halo sits on visible glass and must still be dusk-graded).
    fixture = np.clip(np.maximum.reduce([hood, interior, bulb]), 0, 1)
    fixture = fixture * (1.0 - 0.90 * sstep((Y - 205) / 12.0))
    fixture = cv2.GaussianBlur(np.clip(fixture, 0, 1), (0, 0), 1.2)
    return {
        'hood': np.clip(hood, 0, 1),
        'interior': np.clip(interior, 0, 1),
        'bulb': np.clip(bulb, 0, 1),
        'glow': np.clip(glow, 0, 1),
        'protect': protect,
        'fixture': fixture,
    }


def lamp_color_mask(frame):
    """Cream underside + copper hood from colour, clipped to a loose bowl.
    Catches any part of the opening the geometric ellipse under-covers.
    """
    img = frame.astype(np.float32)
    B, G, R = img[..., 0], img[..., 1], img[..., 2]
    Yv = luma(img)
    ang = np.deg2rad(-17.0)
    dx, dy = X - 323.0, Y - 132.0
    u = dx * np.cos(ang) + dy * np.sin(ang)
    v = -dx * np.sin(ang) + dy * np.cos(ang)
    bowl = ellipse_fill(u, v - 4.0, 140.0, 78.0, 0.14)
    cream = (
        sstep((R - B - 40.0) / 28.0)
        * sstep((G - B - 20.0) / 28.0)
        * sstep((Yv - 90.0) / 20.0)
        * sstep((245.0 - Yv) / 20.0)
    )
    metal = (
        sstep((R - B - 12.0) / 22.0)
        * sstep((Yv - 28.0) / 16.0)
        * sstep((150.0 - Yv) / 28.0)
        * sstep((G - 35.0) / 20.0)
    )
    m = np.clip(np.maximum(cream, metal) * bowl, 0, 1)
    return cv2.GaussianBlur(m, (0, 0), 1.3)


def lamp_pool():
    """Downward cone / radial pool centred on the bulb. 1 at the lamp, 0 far."""
    dx = X - LX
    dy = Y - LY
    # Anisotropic: tighter upward (the shade blocks it), wider down onto the desk
    up = np.clip(-dy, 0, None)
    down = np.clip(dy, 0, None)
    r = np.sqrt((dx / 260.0) ** 2 + (up / 140.0) ** 2 + (down / 380.0) ** 2)
    pool = np.exp(-0.5 * (r / 0.62) ** 2)
    # a secondary, broader desk wash
    desk = np.exp(-0.5 * ((dx / 320.0) ** 2 + ((Y - 520.0) / 160.0) ** 2))
    desk = desk * sstep((Y - 430) / 30.0)
    pool = np.clip(np.maximum(pool, 0.65 * desk), 0, 1)
    return cv2.GaussianBlur(pool, (0, 0), 10.0)


def dusk_plate(w_breath):
    """Smooth vertical late-dusk ramp. No flat mid-sky plateau.

    Open sky (top of the pane, t≈0–0.25) eases from deep navy toward a
    hint of warmth at the treeline. A thin rose-to-gold afterglow sits
    low, behind the trees, then cools again toward the snowy ground.
    """
    t = np.clip((Y - 90.0) / (396.0 - 90.0), 0, 1)
    # Visible ramp already inside the open-sky zone (t≈0–0.25, y 90–165),
    # then a thin afterglow behind the trees. Every step is a small ease —
    # no long hold on one hex.
    upper = hex_bgr('0B102F')
    upper2 = hex_bgr('0F1644')
    mid = hex_bgr('171848')
    blend = hex_bgr('2D1F37')
    rose = hex_bgr('8A4A38') * (0.78 + 0.10 * w_breath)
    rose2 = hex_bgr('C47A6A') * (0.52 + 0.14 * w_breath)
    gold = hex_bgr('D4A565') * (0.42 + 0.12 * w_breath)
    cool_g = hex_bgr('142658')
    cool_g2 = hex_bgr('1E183B')
    stops = np.array(
        [0.00, 0.08, 0.15, 0.22, 0.28, 0.35, 0.43, 0.52, 0.62, 0.75, 0.88, 1.00],
        np.float32,
    )
    cols = np.stack([
        upper,                                          # deep navy at the frame
        upper2,
        mid,
        mid * 0.55 + blend * 0.45,                      # already easing at the treeline
        blend * 0.40 + rose * 0.60,                     # afterglow begins behind treetops
        rose * 0.35 + rose2 * 0.45 + gold * 0.20,       # thin rose-gold lip
        rose2 * 0.20 + gold * 0.10 + blend * 0.30 + cool_g2 * 0.40,
        cool_g2 * 0.70 + blend * 0.30,
        cool_g,
        cool_g * 0.75 + mid * 0.25,
        cool_g2,
        cool_g2,
    ], 0)
    c = np.zeros((H, W, 3), np.float32)
    for k in range(3):
        c[..., k] = np.interp(t, stops, cols[:, k])
    # Slight vertical blur so any residual step is gone; this is a 1-D ramp
    # so a tall kernel only softens the already-smooth interpolation.
    c = cv2.GaussianBlur(c, (0, 0), 8.0)
    return c


def leaf_in_pane(frame, panes):
    """Indoor leaves that overlap the glass (desk plants, hanging ivy)."""
    leaf = C.green_leaf(frame)
    # Only where they sit on/over the pane, plus a small dilation so edges stay
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    dil = cv2.dilate((leaf > 0.25).astype(np.uint8), ker)
    a = np.clip(leaf, 0, 1) * 0.4 + dil.astype(np.float32) * 0.6
    a = a * sstep((panes - 0.05) / 0.15)  # only near glass
    return cv2.GaussianBlur(np.clip(a, 0, 1), (0, 0), 1.6)


def cat_protect(alpha):
    """Dilate+feather the silhouette so dusk cannot leak into fur or ear tufts."""
    a = alpha.astype(np.float32)
    if a.max() > 1.5:
        a = a / 255.0
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))  # ~6 px, covers ear tufts
    dil = cv2.dilate((a > 0.30).astype(np.uint8), ker).astype(np.float32)
    out = np.maximum(a, dil)
    return cv2.GaussianBlur(np.clip(out, 0, 1), (0, 0), 2.2)


def grade_window(frame, panes, cat_p, lamp_fixture, leaves, steam, title, w_breath):
    """Recompose a painted dusk sky under the trees and snow, glass only.

    The dusk plate is the background everywhere in the glass. Open sky
    reveals it fully; trees and ground are a darker, lower-contrast,
    cooler grade of the original, with a faint afterglow bounce, sitting
    on top of the same plate so the sky never reads as a pasted stripe.
    """
    img = frame.astype(np.float32)
    glass = panes * (1.0 - cat_p) * (1.0 - lamp_fixture) * (1.0 - leaves) * (1.0 - 0.7 * steam) * (1.0 - title)
    glass = np.clip(glass, 0, 1)
    if float(glass.max()) < 1e-4:
        return img, glass
    low = cv2.GaussianBlur(img, (0, 0), 5.5)
    residual = img - low
    Ylow = luma(low)
    B, G, R = low[..., 0], low[..., 1], low[..., 2]
    # Open sky: warm pale (R >> B). Cool bright pixels are snow, not sky.
    warm = sstep(((R - B) - 8.0) / 28.0)
    bright = sstep((Ylow - 128.0) / 32.0)
    very_bright = sstep((Ylow - 175.0) / 22.0)
    cool_snow = sstep(((B - R) - 4.0) / 20.0)
    sky = np.clip(bright * warm + very_bright * (1.0 - cool_snow) * 0.25, 0, 1) * glass
    # Heavy feather at the treeline so the sky does not cut as a hard band.
    sky = cv2.GaussianBlur(sky, (0, 0), 7.0)
    sky = np.clip(sky * glass, 0, 1)

    dusk = dusk_plate(w_breath)
    t = np.clip((Y - 90.0) / (396.0 - 90.0), 0, 1)
    # Dusk-lit landscape: darker, cooler, lower contrast than day snow.
    # Upper trees are pulled down further so they do not sit as a bright
    # shelf under the navy sky.
    mid = 85.0
    contrast = 0.32
    Y2 = (22.0 + 4.0 * w_breath) + (Ylow - mid) * contrast
    # Extra exposure drop toward the top of the pane (treetops → sky).
    top_drop = 0.55 + 0.45 * sstep((t - 0.08) / 0.28)
    Y2 = Y2 * top_drop
    scale = np.clip(Y2 / (Ylow + 1e-3), 0.10, 0.50)
    land_cool = np.array([1.12, 0.97, 0.78], np.float32)  # BGR
    land = low * scale[..., None] * land_cool
    # Faint warm afterglow reflection on mid/low snow and lower trees.
    ag = sstep((t - 0.22) / 0.14) * sstep((0.80 - t) / 0.20)
    ag = cv2.GaussianBlur(ag, (0, 0), 12.0)
    bounce = (0.045 + 0.030 * w_breath) * ag * np.clip(Ylow / 110.0, 0.15, 1.2)
    land = land + hex_bgr('C47A6A') * bounce[..., None] + hex_bgr('D4A565') * (0.30 * bounce)[..., None]
    # Recompose: plate is the background. Open sky is almost pure plate.
    # Treetops dissolve into the plate so the sky is not a pasted stripe.
    dissolve = sstep((0.38 - t) / 0.26)
    dissolve = cv2.GaussianBlur(dissolve, (0, 0), 8.0)
    land_a = np.clip((1.0 - sky) * (1.0 - 0.70 * dissolve), 0, 1)
    new_low = dusk * (1.0 - land_a)[..., None] + land * land_a[..., None]
    # Keep falling snow; lift slightly so flakes still read on the darker pane.
    snow_gain = 1.28
    new_pane = new_low + residual * snow_gain
    g = glass[..., None]
    out = img * (1.0 - g) + new_pane * g
    return out, glass


def grade_room(img, pool, lamp, cat, title, ph, glass=None):
    """Darken away from the lamp; warm the lamp and a desk pool. No cool shift on the lamp."""
    protect = lamp['protect']
    bulb = lamp['bulb']
    interior = lamp['interior']
    glow = lamp['glow']
    # 7-cycle flicker only (do not stack a new 3-cycle pulse on the v28 one)
    flick = 1.0 + 0.006 * np.sin(2.0 * np.pi * 7.0 * ph / NCYCLE)

    # Gain: near the lamp ~1, far down to ~0.28 of the current luma.
    # Cat is darkened more gently (face stays readable).
    # Outdoor glass is already dusk-graded; do not re-darken it with the
    # interior pool (that left the left pane bright and the right pane dark).
    far = 1.0 - pool
    cat_soft = cv2.GaussianBlur(cat, (0, 0), 8.0)
    glass_a = np.zeros((H, W), np.float32) if glass is None else glass
    dark = 0.72 * far * (1.0 - 0.92 * protect) * (1.0 - title) * (1.0 - 0.92 * glass_a)
    dark = dark * (1.0 - 0.28 * cat_soft)  # slightly deeper on the cat, face still open
    dark = cv2.GaussianBlur(np.clip(dark, 0, 1), (0, 0), 6.0)
    gain = 1.0 - 0.68 * dark  # far pixels ~0.32x, lamp ~1.0 (plant still reads)
    # A slight cool shift ONLY in the deep far room, never on the lamp or cat face
    cool = np.array([1.06, 0.96, 0.86], np.float32)
    cool_amt = 0.35 * far * far * (1.0 - protect) * (1.0 - 0.85 * cat_soft) * (1.0 - title) * (1.0 - glass_a)
    cool_amt = cv2.GaussianBlur(np.clip(cool_amt, 0, 1), (0, 0), 10.0)
    graded = img * gain[..., None]
    graded = graded * (1.0 - cool_amt)[..., None] + (graded * cool) * cool_amt[..., None]

    # Warm lift in the pool (desk + air), and a stronger warm lift on the fixture itself
    warm = np.array([0.88, 1.00, 1.14], np.float32)  # BGR
    pool_w = pool * (1.0 - title) * (1.0 - 0.85 * glass_a) * flick
    graded = graded * (1.0 + (warm - 1.0) * (0.22 * pool_w)[..., None])

    # Shade interior and bulb: push toward #FFF0C9, never cooler
    core = hex_bgr('FFF0C9')
    bamt = np.clip(0.40 * bulb + 0.16 * interior, 0, 1) * flick
    bamt = cv2.GaussianBlur(bamt, (0, 0), 2.0)
    graded = graded * (1.0 - bamt)[..., None] + (graded * 0.45 + core * 0.55) * bamt[..., None]
    # Mild extra luma on the bulb — keep the disc readable, don't clip to 255
    graded = graded * (1.0 + 0.08 * bulb * flick)[..., None]

    # Warm bloom around the opening, not on the face
    bloom_c = hex_bgr('FFE6B0')
    bloom_m = cv2.GaussianBlur((0.65 * interior + 0.45 * bulb + 0.35 * glow) * (1.0 - 0.85 * cat_soft) * (1.0 - title), (0, 0), 22.0)
    bloom_m2 = cv2.GaussianBlur((0.8 * bulb + 0.4 * interior) * (1.0 - title), (0, 0), 8.0)
    graded = graded + bloom_c * ((0.16 * flick) * bloom_m + (0.10 * flick) * bloom_m2)[..., None]

    # Warm desk pool (additive, small)
    desk = pool * sstep((Y - 450) / 25.0) * sstep((620 - Y) / 20.0) * sstep((X - 80) / 30.0) * sstep((520 - X) / 40.0)
    desk = cv2.GaussianBlur(np.clip(desk, 0, 1), (0, 0), 12.0)
    graded = graded + hex_bgr('E8C090') * (0.16 * desk * flick)[..., None]

    return graded


def apply_grain(img, ph, cat, title):
    # Modest looping grain, lighter on the face. 241-phase, stable.
    rng = np.random.RandomState(41 + ph)
    n = rng.randn(H, W).astype(np.float32)
    n = cv2.GaussianBlur(n, (0, 0), 0.55)
    n = n / (n.std() + 1e-6)
    face = cv2.GaussianBlur(cat, (0, 0), 10.0)
    op = 0.07 * (1.0 - 0.80 * face) * (1.0 - 0.6 * title)
    gcol = n[..., None] * np.array([0.96, 1.00, 1.04], np.float32)
    return img + gcol * (2.4 * op)[..., None]


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


def get_matte(i, cache):
    if cache is not None:
        return cache[i]
    p = MATTE_PNG / f'cat-n{i}.png'
    if not p.exists():
        raise SystemExit(f'missing cat matte for n{i}: {p}')
    return cv2.imread(str(p), 0)


def process(src, mattes, preview=None):
    Title = C.title_mask()
    Steam = C.steam_mask()
    Panes = pane_mask()
    Lamp = lamp_masks()
    Pool = lamp_pool()
    want = set(preview) if preview else set(range(NFR))
    out = [None] * NFR
    for i in range(NFR):
        if i not in want:
            continue
        ph = phase_of(i)
        w = 0.5 - 0.5 * np.cos(2.0 * np.pi * ph / NCYCLE)
        frame = src[i]
        cat = get_matte(i, mattes)
        cat_p = cat_protect(cat)
        cat_a = cat.astype(np.float32)
        if cat_a.max() > 1.5:
            cat_a = cat_a / 255.0
        leaves = leaf_in_pane(frame, Panes)
        fixture = np.clip(np.maximum(Lamp['fixture'], lamp_color_mask(frame)), 0, 1)
        img, glass = grade_window(frame, Panes, cat_p, fixture, leaves, Steam, Title, w)
        # Window grade must never cool the fixture: put the source lamp back,
        # then grade_room applies only the warm lift.
        img = img * (1.0 - fixture)[..., None] + frame.astype(np.float32) * fixture[..., None]
        img = grade_room(img, Pool, Lamp, cat_a, Title, ph, glass=glass)
        img = apply_grain(img, ph, cat_a, Title)
        img = np.clip(img, 0, 255)
        # Approved scrolling title — never retouch
        img[14:60, 16:165] = frame[14:60, 16:165]
        out[i] = img.astype(np.uint8)
        if i % 25 == 0 or i in (0, 60, 120, 138, 180, 370, 432):
            Yi = luma(out[i])
            print(
                f'  n{i} ph={ph} glob={Yi.mean():.1f} '
                f'bulb={Yi[130:165,305:340].mean():.1f} '
                f'interior={Yi[110:160,260:380].mean():.1f} '
                f'face={Yi[260:430,600:820].mean():.1f} '
                f'win={Yi[100:180,300:500].mean():.1f} '
                f'far={Yi[80:300,1100:1270].mean():.1f} '
                f'desk={Yi[480:600,200:500].mean():.1f} '
                f'ratio={Yi[110:160,260:380].mean() / max(Yi[80:300,1100:1270].mean(), 1):.2f}',
                flush=True,
            )
    return out


def save_preview(src, out, idx):
    q = QC / 'preview'
    tiles = []
    for i in idx:
        if out[i] is None:
            continue
        cv2.imwrite(str(q / f'full-n{i}.png'), out[i])
        cv2.imwrite(str(q / f'full-n{i}.jpg'), out[i], [int(cv2.IMWRITE_JPEG_QUALITY), 93])
        # side-by-side vs v41d
        ba = np.hstack([src[i], out[i]])
        cv2.imwrite(str(q / f'ba-n{i}.jpg'), ba, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        # crops
        def crop(tag, box):
            y0, y1, x0, x1 = box
            a, b = src[i][y0:y1, x0:x1], out[i][y0:y1, x0:x1]
            # 2x
            a2 = cv2.resize(a, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
            b2 = cv2.resize(b, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(str(q / f'{tag}-n{i}.jpg'), np.hstack([a2, b2]), [int(cv2.IMWRITE_JPEG_QUALITY), 93])
        crop('window', (70, 420, 210, 850))
        crop('face', (220, 480, 540, 900))
        crop('ears', (130, 310, 500, 900))
        crop('lamp', (50, 280, 140, 460))
        crop('mouth', (340, 500, 600, 820))
        crop('plant', (430, 720, 1000, 1279))
        # luma map
        Y = luma(out[i])
        heat = cv2.applyColorMap(np.clip(Y, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        cv2.imwrite(str(q / f'luma-n{i}.jpg'), np.hstack([out[i], heat]), [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        # contact thumb
        th = cv2.resize(out[i], (320, 180), interpolation=cv2.INTER_AREA)
        cv2.putText(th, f'n{i}', (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(th, f'n{i}', (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        tiles.append(th)
    if tiles:
        # 4 + rest
        row1 = np.hstack(tiles[:4]) if len(tiles) >= 4 else np.hstack(tiles)
        if len(tiles) > 4:
            pad = [np.zeros_like(tiles[0])] * (4 - (len(tiles) - 4) % 4 if (len(tiles) - 4) % 4 else 0)
            row2 = np.hstack(tiles[4:] + pad) if pad else np.hstack(tiles[4:])
            if row2.shape[1] < row1.shape[1]:
                row2 = np.pad(row2, ((0, 0), (0, row1.shape[1] - row2.shape[1]), (0, 0)))
            contact = np.vstack([row1, row2])
        else:
            contact = row1
        cv2.imwrite(str(q / 'contact-sheet.jpg'), contact, [int(cv2.IMWRITE_JPEG_QUALITY), 93])
        print('preview + contact written', q)


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
        ('pcm', PCM, PCM_SHA),
    ):
        got = sha256(path)
        if got != expect:
            raise SystemExit(f'{label} sha changed: {got}')
    print('frozen shas ok', flush=True)

    print('loading v41d lossless…', flush=True)
    src = load_span(SRC)
    assert len(src) == NFR

    mattes = None
    if MATTE.exists():
        print('loading cat mattes…', flush=True)
        mattes = load_span(MATTE, gray=True)
        assert len(mattes) == NFR, len(mattes)

    if args.preview:
        idx = [0, 60, 120, 138, 180, 241, 370, 432]
        out = process(src, mattes, preview=idx)
        save_preview(src, out, idx)
        return

    if mattes is None:
        raise SystemExit(f'cat mattes missing: {MATTE}')
    out = process(src, mattes)
    bad = [i for i, (a, b) in enumerate(zip(src, out))
           if not np.array_equal(a[14:60, 16:165], b[14:60, 16:165])]
    print('title mismatches' if bad else 'title core byte-identical', len(bad) if bad else '')
    los = WORK / 'v41f-r2-span-890-915-lossless.mkv'
    mp4 = P / 'v41f-r2-test-25s.mp4'
    print('writing lossless…', flush=True)
    write_lossless(out, los)
    print('encoding mp4…', flush=True)
    encode_mp4(out, mp4)
    print('v41f-r2 mp4', sha256(mp4))
    print('v41 still', sha256(V41_MP4))
    print('v41d still', sha256(V41D_MP4))
    print('v41e still', sha256(V41E_MP4))
    print('v41e-r2 still', sha256(V41ER2_MP4))
    print('v41f still', sha256(V41F_MP4))

if __name__ == '__main__':
    main()
