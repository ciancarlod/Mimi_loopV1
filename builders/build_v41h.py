#!/usr/bin/env python3
"""v41h: painted tail + corrected dusk sky ramp.

Keeps v41 / v41b / v41c / v41d / v41e / v41e-r2 / v41f / v41f-r2 / v41g frozen.

Source is v41d lossless (same as the atmosphere passes) so the window can
be re-graded. Room grade, lamp restore and grain follow the locked v41f-r2
look. Dust and the right-corner plant brush come from v41g. The v41g
haunch-curl is dropped; a real fluffy tail is composited and spline-animated.
"""
from __future__ import annotations
import hashlib, subprocess as sp, sys
from pathlib import Path
import cv2
import numpy as np

P = Path('/workspace/lofi-v41-meow')
QC = P / 'qc' / 'v41h'
QC.mkdir(parents=True, exist_ok=True)
(QC / 'preview').mkdir(exist_ok=True)
(QC / 'encoded').mkdir(exist_ok=True)
WORK = P / 'v41h'
WORK.mkdir(exist_ok=True)

SRC = P / 'v41d' / 'v41d-span-890-915-lossless.mkv'
PCM = P / 'work' / 'test-audio-890-915-v41.f32le'
MATTE = P / 'v41f' / 'cat-alpha-890-915.mkv'
SPRITE = WORK / 'tail-sprite.png'

V41_MP4 = P / 'v41-test-25s.mp4'
V41D_MP4 = P / 'v41d-test-25s.mp4'
V41E_MP4 = P / 'v41e-test-25s.mp4'
V41ER2_MP4 = P / 'v41e-r2-test-25s.mp4'
V41F_MP4 = P / 'v41f-test-25s.mp4'
V41FR2_MP4 = P / 'v41f-r2-test-25s.mp4'
V41G_MP4 = P / 'v41g-test-25s.mp4'

V41_SHA = 'f53cb1127c75775b740b555dd856e0002c784f80a2a3097118d8e197a794e95f'
V41D_SHA = 'd1dab25187f4756969344036f50cd0ef89fac7c414d80006b8853acf04be2b34'
V41E_SHA = '211f5bba5d04ae55234ebedb5d7b1ce1b7b9e2abedcdccdcc54e6cce827fd724'
V41ER2_SHA = 'bc19246dc513443cb78cc15a9ff4054be5fa66a73b59740a8c5c31bae74986ff'
V41F_SHA = 'b697e8b6e223e94a0e26a97abeb85b29764c70a35037680eed6b936e138ca703'
V41FR2_SHA = 'e7db48cc7d147a9e8ed86568c4583dc7bf1559911ff98b846194b2795171ab50'
V41G_SHA = '75716001220f0222f1f382a6b73f23e117e2e642a9cd17b751322f44a09d87bb'
PCM_SHA = 'a635074f164c14fb3faabbf33c258703e9b7b9645eddbf8028f63a29573745a0'
NFR, W, H, FPS, NCYCLE = 600, 1280, 720, 24, 241
TITLE = (14, 60, 16, 165)

sys.path.insert(0, str(P))
import build_v41c as C
import build_v41f_r2 as R
import build_v41g as G

# Placement B: scale 0.52, base tucked behind the haunch, tip among mid leaves.
TAIL_SCALE = 0.52
BASE_W = np.array([970.0, 660.0], np.float32)
TIP_W = np.array([1170.0, 575.0], np.float32)


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
    return R.load_span(path, gray=gray)


def encode_mp4(frames, path):
    R.encode_mp4(frames, path)


def write_lossless(frames, path):
    R.write_lossless(frames, path)


Y, X = R.Y, R.X


# ---------- corrected dusk plate + window grade ----------
def dusk_plate(w_breath):
    """Darkest at the top of the pane, lighter and warmer toward the treetops."""
    t = np.clip((Y - 90.0) / (396.0 - 90.0), 0, 1)
    upper = hex_bgr('0C1238')
    midn = hex_bgr('1A1F4A')
    blue = hex_bgr('2E3A5C')
    blue2 = hex_bgr('3A4868')
    blend = hex_bgr('3A2A40')
    rose = hex_bgr('8A4A38') * (0.58 + 0.12 * w_breath)
    gold = hex_bgr('D4A565') * (0.36 + 0.10 * w_breath)
    cool = hex_bgr('142658')
    cool2 = hex_bgr('1E183B')
    stops = np.array([0.00, 0.06, 0.13, 0.20, 0.25, 0.31, 0.38, 0.48, 0.62, 0.80, 1.00], np.float32)
    cols = np.stack([
        upper,
        upper * 0.55 + midn * 0.45,
        midn,
        midn * 0.30 + blue * 0.70,
        blue * 0.40 + blue2 * 0.30 + rose * 0.30,          # afterglow begins just above treetops
        blue2 * 0.22 + rose * 0.45 + gold * 0.33,            # thin rose-gold lip
        rose * 0.12 + gold * 0.08 + blend * 0.25 + cool * 0.55,
        cool,
        cool * 0.6 + cool2 * 0.4,
        cool2,
        cool2,
    ], 0)
    c = np.zeros((H, W, 3), np.float32)
    for k in range(3):
        c[..., k] = np.interp(t, stops, cols[:, k])
    return cv2.GaussianBlur(c, (0, 0), 7.0), t


def grade_window(frame, panes, cat_p, lamp_fixture, leaves, steam, title, w_breath):
    glass = panes * (1.0 - cat_p) * (1.0 - lamp_fixture) * (1.0 - leaves) * (1.0 - 0.7 * steam) * (1.0 - title)
    glass = np.clip(glass, 0, 1)
    if float(glass.max()) < 1e-4:
        return frame.astype(np.float32), glass
    img = frame.astype(np.float32)
    dusk, t = dusk_plate(w_breath)
    low = cv2.GaussianBlur(img, (0, 0), 4.0)
    Ylow = luma(low)
    Yf = luma(img)
    mu = cv2.GaussianBlur(Yf, (0, 0), 3.5)
    var = cv2.GaussianBlur((Yf - mu) ** 2, (0, 0), 3.5)
    textured = sstep((np.sqrt(np.clip(var, 0, None)) - 6.0) / 8.0)
    upper = sstep((0.34 - t) / 0.10)
    bright = sstep((Ylow - 140.0) / 35.0)
    sky = np.clip(upper * bright * (1.0 - 0.75 * textured), 0, 1)
    top = sstep((0.14 - t) / 0.08)
    tree = np.clip(upper * textured * sstep((170.0 - Ylow) / 40.0), 0, 1)
    sky = np.clip(np.maximum(sky, top * (1.0 - 0.85 * tree)), 0, 1) * glass
    sky = np.clip(cv2.GaussianBlur(sky, (0, 0), 4.5) * glass, 0, 1)
    tree = cv2.GaussianBlur(tree * glass, (0, 0), 3.0)

    # Landscape: darker than the horizon plate so trees silhouette against it.
    Y2 = (18.0 + 3.0 * w_breath) + (Ylow - 90.0) * 0.26
    sil = sstep((0.42 - t) / 0.22) * sstep((t - 0.08) / 0.12)
    Y2 = Y2 * (1.0 - 0.48 * sil * np.clip(tree + 0.40, 0, 1))
    Y2 = np.clip(Y2, 8, 48)
    scale = np.clip(Y2 / (Ylow + 1e-3), 0.08, 0.42)
    land = low * scale[..., None] * np.array([1.10, 0.98, 0.80], np.float32)
    ag = sstep((t - 0.18) / 0.10) * sstep((0.70 - t) / 0.22)
    ag = cv2.GaussianBlur(ag, (0, 0), 10.0)
    bounce = (0.048 + 0.028 * w_breath) * ag * np.clip(Ylow / 120.0, 0.1, 1.0)
    land = land + hex_bgr('C47A6A') * bounce[..., None] + hex_bgr('D4A565') * (0.30 * bounce)[..., None]
    land_a = np.clip((1.0 - sky) * glass, 0, 1)
    new_low = dusk * (1.0 - land_a)[..., None] + land * land_a[..., None]

    # Snowflakes: small-scale tophat only, capped, so the old bright sky
    # cannot punch through the plate as mid-frequency leftover.
    Y8 = Yf.astype(np.uint8)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    hat = cv2.morphologyEx(Y8, cv2.MORPH_TOPHAT, ker).astype(np.float32)
    hat = np.clip(hat, 0, 22.0)
    flake = np.array([1.06, 1.00, 0.92], np.float32)
    # slightly more flake in the sky so they read on navy, less on already-bright snow
    snow_amt = (1.15 * (1.0 - 0.45 * land_a) + 0.55 * land_a)
    new_pane = new_low + hat[..., None] * flake * snow_amt[..., None]
    new_pane = np.clip(new_pane, 0, 255)
    g = glass[..., None]
    out = img * (1.0 - g) + new_pane * g
    return out, glass


# ---------- tail sprite, centerline, spline deform ----------
def _trace_ridge(dt, start, end):
    p = np.array(start, np.int32)
    end = np.array(end, np.int32)
    h, w = dt.shape
    path = [p.copy()]
    seen = set()
    for _ in range(2500):
        seen.add((int(p[0]), int(p[1])))
        if np.hypot(*(end - p)) <= 3:
            path.append(end.copy())
            break
        best, bests = None, -1e18
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if dx == 0 and dy == 0:
                    continue
                q = p + np.array([dx, dy])
                x, y = int(q[0]), int(q[1])
                if x < 0 or y < 0 or x >= w or y >= h:
                    continue
                if (x, y) in seen:
                    continue
                if dt[y, x] < 0.8:
                    continue
                sc = dt[y, x] * 2.0 - 0.08 * np.hypot(*(end - q))
                if sc > bests:
                    bests, best = sc, q
        if best is None:
            # fall back to a step toward the end
            d = end - p
            n = np.hypot(*d) + 1e-6
            best = (p + np.round(d / n)).astype(np.int32)
        p = best.astype(np.int32)
        if len(path) == 1 or not np.array_equal(p, path[-1]):
            path.append(p.copy())
    return np.array(path, np.float32)


def _resample_poly(poly, n=18):
    d = np.sqrt(((poly[1:] - poly[:-1]) ** 2).sum(1))
    arc = np.concatenate([[0], np.cumsum(d)])
    if arc[-1] < 1:
        return np.linspace(poly[0], poly[-1], n)
    s = np.linspace(0, arc[-1], n)
    x = np.interp(s, arc, poly[:, 0])
    y = np.interp(s, arc, poly[:, 1])
    return np.stack([x, y], 1).astype(np.float32)


def _arc(poly):
    d = np.sqrt(((poly[1:] - poly[:-1]) ** 2).sum(1))
    arc = np.concatenate([[0], np.cumsum(d)])
    return arc, float(arc[-1])


def _closest(px, py, poly, arc, total):
    """Vectorized closest point, s in [0,1], signed distance, tangent."""
    best_d2 = np.full(px.shape, 1e18, np.float32)
    best_s = np.zeros(px.shape, np.float32)
    best_cx = np.zeros(px.shape, np.float32)
    best_cy = np.zeros(px.shape, np.float32)
    best_tx = np.ones(px.shape, np.float32)
    best_ty = np.zeros(px.shape, np.float32)
    for i in range(len(poly) - 1):
        ax, ay = poly[i]
        bx, by = poly[i + 1]
        vx, vy = bx - ax, by - ay
        L2 = vx * vx + vy * vy + 1e-8
        t = np.clip(((px - ax) * vx + (py - ay) * vy) / L2, 0, 1)
        cx, cy = ax + t * vx, ay + t * vy
        d2 = (px - cx) ** 2 + (py - cy) ** 2
        m = d2 < best_d2
        best_d2 = np.where(m, d2, best_d2)
        best_s = np.where(m, (arc[i] + t * np.sqrt(L2)) / (total + 1e-8), best_s)
        best_cx = np.where(m, cx, best_cx)
        best_cy = np.where(m, cy, best_cy)
        ln = np.sqrt(L2)
        best_tx = np.where(m, vx / ln, best_tx)
        best_ty = np.where(m, vy / ln, best_ty)
    # signed dist: cross(tangent, r)
    sd = (px - best_cx) * (-best_ty) + (py - best_cy) * best_tx  # wait, normal is (-ty, tx)? 
    # tangent (tx,ty), +normal ( -ty, tx ) or (ty, -tx). Use cross z = tx*(py-cy) - ty*(px-cx)
    sd = best_tx * (py - best_cy) - best_ty * (px - best_cx)
    return best_s, sd, best_tx, best_ty


def _prep_tail():
    sp = cv2.imread(str(SPRITE), cv2.IMREAD_UNCHANGED)
    assert sp is not None and sp.shape[2] == 4, SPRITE
    a = sp[:, :, 3].astype(np.float32) / 255.0
    rgb = sp[:, :, :3].astype(np.float32)
    # extra settle into the dusk room
    rgb = rgb * 0.90
    # Hand-traced medial axis on tail-sprite.png (360x384). The automatic
    # ridge walk folded back on itself and turned the warp into a starburst.
    poly_s = np.array([
        [25, 305], [40, 255], [55, 210], [75, 170], [110, 130],
        [155, 100], [200,  78], [245,  68], [285,  78], [312, 108],
    ], np.float32)
    base = poly_s[0].copy()
    tip = poly_s[-1].copy()

    # scale
    sh, sw = sp.shape[:2]
    nh, nw = int(round(sh * TAIL_SCALE)), int(round(sw * TAIL_SCALE))
    rgb_s = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    a_s = cv2.resize(a, (nw, nh), interpolation=cv2.INTER_AREA)
    poly_s = poly_s * TAIL_SCALE
    base_s = base * TAIL_SCALE
    tip_s = tip * TAIL_SCALE

    # rotate around base so rest tip aims at TIP_W
    v_src = tip_s - base_s
    v_dst = TIP_W - BASE_W
    ang = np.degrees(np.arctan2(v_dst[1], v_dst[0]) - np.arctan2(v_src[1], v_src[0]))
    M = cv2.getRotationMatrix2D(tuple(base_s), -ang, 1.0)
    rgb_r = cv2.warpAffine(rgb_s, M, (nw, nh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    a_r = cv2.warpAffine(a_s, M, (nw, nh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    # transform polyline
    ones = np.ones((len(poly_s), 1), np.float32)
    pr = (M @ np.hstack([poly_s, ones]).T).T
    br = (M @ np.array([base_s[0], base_s[1], 1.0])).ravel()
    # world translation
    ox, oy = BASE_W[0] - br[0], BASE_W[1] - br[1]
    rest_poly = pr + np.array([ox, oy], np.float32)
    # Half-width along the rest poly, in world/sprite-local pixels (scale already applied).
    dt = cv2.distanceTransform((a_r > 0.25).astype(np.uint8), cv2.DIST_L2, 3)
    widths = []
    for q in pr:  # sprite-local after rotation
        x, y = int(round(q[0])), int(round(q[1]))
        x = int(np.clip(x, 0, nw - 1)); y = int(np.clip(y, 0, nh - 1))
        widths.append(float(dt[y, x]))
    widths = np.array(widths, np.float32)
    widths = np.maximum(widths, 8.0)
    # World-space rest sprite (identity composite when undeformed).
    world_rgb = np.zeros((H, W, 3), np.float32)
    world_a = np.zeros((H, W), np.float32)
    x0, y0 = int(np.floor(ox)), int(np.floor(oy))
    # rgb_r is already in a canvas of size (nh,nw) whose (0,0) is sprite-local
    # world pos = sprite-local + (ox,oy)
    # use rounded ox,oy — subpixel is handled by having ox,oy floats on the poly only
    def _paste(dst, src, x0, y0):
        h, w = src.shape[:2]
        y1, x1 = y0 + h, x0 + w
        sy0 = sx0 = 0
        sy1, sx1 = h, w
        if y0 < 0:
            sy0 = -y0; y0 = 0
        if x0 < 0:
            sx0 = -x0; x0 = 0
        if y1 > dst.shape[0]:
            sy1 -= (y1 - dst.shape[0]); y1 = dst.shape[0]
        if x1 > dst.shape[1]:
            sx1 -= (x1 - dst.shape[1]); x1 = dst.shape[1]
        if y1 > y0 and x1 > x0:
            dst[y0:y1, x0:x1] = src[sy0:sy1, sx0:sx1]
    _paste(world_rgb, rgb_r, int(round(ox)), int(round(oy)))
    _paste(world_a, a_r, int(round(ox)), int(round(oy)))
    return {
        'rgb': rgb_r, 'a': a_r, 'ox': ox, 'oy': oy,
        'rest': rest_poly, 'nw': nw, 'nh': nh,
        'widths': widths,
        'world_rgb': world_rgb, 'world_a': world_a,
    }


def _deform_poly(rest, ph):
    """Tip-led idle + one larger curl per 241-cycle. Base stays put."""
    poly = rest.copy()
    arc, total = _arc(rest)
    s = arc / (total + 1e-8)
    # tangents
    tg = np.zeros_like(poly)
    tg[1:-1] = poly[2:] - poly[:-2]
    tg[0] = poly[1] - poly[0]
    tg[-1] = poly[-1] - poly[-2]
    ln = np.sqrt((tg ** 2).sum(1, keepdims=True)) + 1e-8
    tg = tg / ln
    nrm = np.stack([-tg[:, 1], tg[:, 0]], 1)  # left normal
    # toward-plant at the tip (from last rest segment, roughly)
    plant_dir = (TIP_W - BASE_W).astype(np.float32)
    plant_dir = plant_dir / (np.hypot(*plant_dir) + 1e-8)
    # idle: 2-cycle + 5-cycle, mid 6px, tip 13px
    two = np.sin(2 * np.pi * 2 * ph / NCYCLE)
    five = np.sin(2 * np.pi * 5 * ph / NCYCLE + 0.4)
    mid = 6.0 * two
    tip = 13.0 * two + 4.5 * five
    # amplitude grows with s (tip-led): 0 at base, mid at 0.55, tip at 1
    amp = mid * np.clip(s / 0.55, 0, 1) * (s < 0.55) + (
        mid + (tip - mid) * sstep((s - 0.55) / 0.45)
    ) * (s >= 0.55)
    # curl: +10 px at the tip toward the plant, s**1.7, ph 90-130
    curl = 0.0
    if 90.0 < ph < 130.0:
        u = (ph - 90.0) / 40.0
        curl = (0.5 - 0.5 * np.cos(2 * np.pi * u)) * 10.0
    off = nrm * amp[:, None]
    # curl mostly along plant_dir (into the leaves), mixed with normal
    off = off + plant_dir[None, :] * (curl * (s ** 1.7))[:, None]
    poly = poly + off
    poly[0] = rest[0]  # lock the base
    return poly


def _mic_mask():
    mic = np.zeros((H, W), np.float32)
    cv2.ellipse(mic, (1075, 428), (125, 82), -18, 0, 360, 1, -1)
    return cv2.GaussianBlur(mic, (0, 0), 3.5)


def apply_painted_tail(frame, cat_a, ph, tail, mic):
    """Translate cross-sections of the rest-pose world sprite along the spine.

    When the deform is zero this is an identity paste (the known-good rest
    pose). Idle/curl only slide each slice by a few pixels, so the hook cannot
    smear into a rectangle.
    """
    rest = tail['rest']
    defp = _deform_poly(rest, ph)
    arc_r, tot_r = _arc(rest)
    pad = 40
    xs = np.concatenate([rest[:, 0], defp[:, 0]])
    ys = np.concatenate([rest[:, 1], defp[:, 1]])
    x0, x1 = int(max(0, xs.min() - pad)), int(min(W, xs.max() + pad + 1))
    y0, y1 = int(max(0, ys.min() - pad)), int(min(H, ys.max() + pad + 1))
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    # Closest point on the REST spine (not the deformed one).
    s, sd, tx, ty = _closest(xx, yy, rest, arc_r, tot_r)
    w_s = np.interp(np.clip(s, 0, 1) * tot_r, arc_r, tail['widths'])
    tube = sstep((w_s + 10.0 - np.abs(sd)) / 3.5)
    # Slice translation: dest has moved by (C_def - C_rest) at this s,
    # so it should sample the rest sprite at dest - that offset.
    defx = np.interp(s * tot_r, arc_r, defp[:, 0])
    defy = np.interp(s * tot_r, arc_r, defp[:, 1])
    resx = np.interp(s * tot_r, arc_r, rest[:, 0])
    resy = np.interp(s * tot_r, arc_r, rest[:, 1])
    mapx = (xx - (defx - resx)).astype(np.float32)
    mapy = (yy - (defy - resy)).astype(np.float32)
    rgb = cv2.remap(tail['world_rgb'], mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    al = cv2.remap(tail['world_a'], mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    # The world sprite already carries a clean alpha; a tight tube mask was
    # clipping the fat hook. Keep a very loose one only as a safety net.
    al = np.clip(al * sstep((w_s + 28.0 - np.abs(sd)) / 6.0), 0, 1)
    la_full = np.zeros((H, W), np.float32)
    layer = np.zeros((H, W, 3), np.float32)
    la_full[y0:y1, x0:x1] = al
    layer[y0:y1, x0:x1] = rgb
    sh = cv2.GaussianBlur(la_full, (0, 0), 8.0)
    base_w = np.exp(-0.5 * ((X - BASE_W[0]) ** 2 + (Y - BASE_W[1]) ** 2) / (48.0 ** 2))
    shadow = 0.35 * sh * base_w * (1.0 - la_full) * (1.0 - np.clip(cat_a, 0, 1))
    img = frame.astype(np.float32)
    comp = img * (1.0 - 0.75 * shadow[..., None])
    comp = comp * (1.0 - la_full[..., None]) + layer * la_full[..., None]
    cat_s = cv2.GaussianBlur(np.clip(cat_a, 0, 1), (0, 0), 0.8)
    protect = np.clip(np.maximum(cat_s, mic), 0, 1)
    comp = img * protect[..., None] + comp * (1.0 - protect[..., None])
    return np.clip(comp, 0, 255).astype(np.uint8), defp[-1]



def process(src, mattes, plant_plate, face, panes, tail, mic, preview=None):
    Title = C.title_mask()
    Steam = C.steam_mask()
    Panes = R.pane_mask()
    Lamp = R.lamp_masks()
    Pool = R.lamp_pool()
    want = set(preview) if preview else set(range(NFR))
    out = [None] * NFR
    for i in range(NFR):
        if i not in want:
            continue
        ph = phase_of(i)
        w = 0.5 - 0.5 * np.cos(2.0 * np.pi * ph / NCYCLE)
        frame = src[i]
        cat = mattes[i]
        cat_p = R.cat_protect(cat)
        cat_a = cat.astype(np.float32)
        if cat_a.max() > 1.5:
            cat_a = cat_a / 255.0
        leaves = R.leaf_in_pane(frame, Panes)
        fixture = np.clip(np.maximum(Lamp['fixture'], R.lamp_color_mask(frame)), 0, 1)
        img, glass = grade_window(frame, Panes, cat_p, fixture, leaves, Steam, Title, w)
        img = img * (1.0 - fixture)[..., None] + frame.astype(np.float32) * fixture[..., None]
        img = R.grade_room(img, Pool, Lamp, cat_a, Title, ph, glass=glass)
        img = R.apply_grain(img, ph, cat_a, Title)
        img = np.clip(img, 0, 255).astype(np.uint8)
        # dust (room), then plant, then painted tail (in front of chair/leaves, behind cat/mic)
        img = G.apply_dust(img, cat_a, ph, face, panes)
        img = G.apply_plant(img, plant_plate, cat_a, ph)
        img, tip = apply_painted_tail(img, cat_a, ph, tail, mic)
        img = img.copy()
        y0, y1, x0, x1 = TITLE
        img[y0:y1, x0:x1] = frame[y0:y1, x0:x1]
        out[i] = img
        if preview or i % 50 == 0:
            Yi = luma(img)
            print(f'  n{i} ph{ph} glob={Yi.mean():.1f} tip=({tip[0]:.0f},{tip[1]:.0f}) '
                  f'plant={G.plant_amp(ph):.2f} skyL={Yi[110,400]:.0f} skyR={Yi[110,720]:.0f}',
                  flush=True)
    return out


def contact_sheet(frames, path, labels):
    thumbs, labs = [], []
    for f, lab in zip(frames, labels):
        t = cv2.resize(f, (640, 360), interpolation=cv2.INTER_AREA)
        cv2.putText(t, lab, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2, cv2.LINE_AA)
        thumbs.append(t)
        labs.append(lab)
    cols = 4
    rows = int(np.ceil(len(thumbs) / cols))
    while len(thumbs) < rows * cols:
        thumbs.append(np.zeros((360, 640, 3), np.uint8))
    grid = np.vstack([np.hstack(thumbs[r * cols:(r + 1) * cols]) for r in range(rows)])
    cv2.imwrite(str(path), grid, [int(cv2.IMWRITE_JPEG_QUALITY), 92])


def save_preview(src, out, idx):
    prev = QC / 'preview'
    crops = {
        'tail': (480, 720, 920, 1280),
        'plant': (430, 720, 1000, 1280),
        'window': (80, 410, 250, 850),
        'skyL': (90, 200, 270, 560),
        'skyR': (90, 200, 600, 830),
        'face': (200, 520, 520, 900),
        'lamp': (40, 280, 200, 500),
        'mouth': (340, 500, 600, 820),
        'desk': (300, 620, 200, 700),
    }
    sheet, labs = [], []
    for i in idx:
        ph = phase_of(i)
        cv2.imwrite(str(prev / f'full-n{i}.png'), out[i])
        for tag, (y0, y1, x0, x1) in crops.items():
            c = out[i][y0:y1, x0:x1]
            c2 = cv2.resize(c, (c.shape[1] * 2, c.shape[0] * 2), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(str(prev / f'{tag}-n{i}.jpg'), c2, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        c3 = cv2.resize(out[i][500:720, 980:1280], (0, 0), fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(prev / f'tail-3x-n{i}.jpg'), c3, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        c3s = cv2.resize(out[i][90:200, 260:830], (0, 0), fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(prev / f'sky-3x-n{i}.jpg'), c3s, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        sheet.append(out[i])
        labs.append(f'n{i} ph{ph}')
    contact_sheet(sheet, prev / 'contact-sheet.jpg', labs)
    if 0 in idx and 241 in idx:
        a, b = out[0], out[241]
        d = cv2.absdiff(a, b).astype(np.float32)
        print(f'  n0 vs n241 global={d.mean():.4f} tail={d[500:720,960:1280].mean():.4f} '
              f'win={d[90:400,266:830].mean():.4f}')
        pair = np.hstack([a[500:720, 900:1280], b[500:720, 900:1280],
                          np.clip(d[500:720, 900:1280] * 8, 0, 255).astype(np.uint8)])
        cv2.imwrite(str(prev / 'seam-n0-n241-tail.jpg'), pair)
    if 88 in idx and 89 in idx:
        a, b = out[88], out[89]
        d = cv2.absdiff(a, b).astype(np.float32)
        print(f'  n88 vs n89 global={d.mean():.4f} tail={d[500:720,960:1280].mean():.4f} '
              f'luma {luma(a).mean():.2f}->{luma(b).mean():.2f}')
        pair = np.hstack([a[500:720, 900:1280], b[500:720, 900:1280],
                          np.clip(d[500:720, 900:1280] * 8, 0, 255).astype(np.uint8)])
        cv2.imwrite(str(prev / 'seam-n88-n89-tail.jpg'), pair)
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
        ('v41g', V41G_MP4, V41G_SHA),
        ('pcm', PCM, PCM_SHA),
    ):
        got = sha256(path)
        if got != expect:
            raise SystemExit(f'{label} sha changed: {got}')
    print('frozen shas ok', flush=True)
    if not SPRITE.exists():
        raise SystemExit(f'missing tail sprite {SPRITE}')

    print('loading v41d lossless…', flush=True)
    src = load_span(SRC)
    assert len(src) == NFR
    print('loading cat mattes…', flush=True)
    mattes = load_span(MATTE, gray=True)
    assert len(mattes) == NFR
    a0 = mattes[0].astype(np.float32)
    if a0.max() > 1.5:
        a0 = a0 / 255.0
    plant_plate = G.build_plant_plate(src[0], a0)
    face = G.face_protect()
    panes = G.pane_solid()
    print('preparing tail sprite…', flush=True)
    tail = _prep_tail()
    print('  rest poly', tail['rest'].round(1), flush=True)
    mic = _mic_mask()

    if args.preview:
        idx = [0, 60, 88, 89, 120, 138, 180, 189, 199, 219, 241, 370, 432, 440]
        out = process(src, mattes, plant_plate, face, panes, tail, mic, preview=idx)
        save_preview(src, out, idx)
        return

    out = process(src, mattes, plant_plate, face, panes, tail, mic)
    y0, y1, x0, x1 = TITLE
    bad = [i for i, (a, b) in enumerate(zip(src, out))
           if not np.array_equal(a[y0:y1, x0:x1], b[y0:y1, x0:x1])]
    print('title mismatches' if bad else 'title core byte-identical', len(bad) if bad else '')
    los = WORK / 'v41h-span-890-915-lossless.mkv'
    mp4 = P / 'v41h-test-25s.mp4'
    print('writing lossless…', flush=True)
    write_lossless(out, los)
    print('encoding mp4…', flush=True)
    encode_mp4(out, mp4)
    print('v41h mp4', sha256(mp4))
    print('v41h lossless', sha256(los))
    for label, path in (('v41', V41_MP4), ('v41d', V41D_MP4), ('v41e', V41E_MP4),
                        ('v41e-r2', V41ER2_MP4), ('v41f', V41F_MP4),
                        ('v41f-r2', V41FR2_MP4), ('v41g', V41G_MP4)):
        print(label, 'still', sha256(path))


if __name__ == '__main__':
    main()
