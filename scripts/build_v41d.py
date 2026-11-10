#!/usr/bin/env python3
"""v41d: defect + mouth-quality pass on frozen v41c.

Does not overwrite v41, v41b-r1, v41b-r2, or v41c masters.
Audio PCM is the approved v41 stem (mux only).
Atmosphere (dusk, lamp-as-key, flicker, breath, dust, tail, plant, grain)
is not applied here; that is v41e+.
"""
from __future__ import annotations
import hashlib, json, subprocess as sp, sys
from pathlib import Path
import cv2
import numpy as np

P = Path('/workspace/lofi-v41-meow')
QC = P / 'qc' / 'v41d'
QC.mkdir(parents=True, exist_ok=True)
(QC / 'preview').mkdir(exist_ok=True)
(QC / 'encoded').mkdir(exist_ok=True)
WORK = P / 'v41d'
WORK.mkdir(exist_ok=True)

SRC = P / 'v41c' / 'v41c-span-890-915-lossless-frozen.mkv'
PCM = P / 'work' / 'test-audio-890-915-v41.f32le'
V41_MP4 = P / 'v41-test-25s.mp4'
V41B_R2 = P / 'v41b-r2-test-25s.mp4'
V41C_MP4 = P / 'v41c-test-25s.mp4'
V41_SHA = 'f53cb1127c75775b740b555dd856e0002c784f80a2a3097118d8e197a794e95f'
R2_SHA = 'c1578db5782025d9236267c8c852a011e5c6e739ff82f785e62aa392d6f23b0f'
V41C_SHA = 'a752e0545cf8886fde322490c25da5bb45f9d715e0dfda312959bfb99bb29fb9'
PCM_SHA = 'a635074f164c14fb3faabbf33c258703e9b7b9645eddbf8028f63a29573745a0'
NFR, W, H, FPS, NCYCLE = 600, 1280, 720, 24, 241

sys.path.insert(0, str(P))
import build_v41_mouth as M
import build_v41c as C

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

def load_span(path):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    assert len(frames) == NFR, len(frames)
    return frames

Y, X = np.mgrid[:H, :W].astype(np.float32)

# More visible philtrum lift / lip part (upper lip was almost static).
M.SHAPES['medium']['LIFT_U'] = 16.8
M.SHAPES['medium']['lip_part'] = 2.0
M.SHAPES['medium']['head'] = 3.2
M.SHAPES['medium']['chest'] = 3.0
M.SHAPES['big']['LIFT_U'] = 21.2
M.SHAPES['big']['lip_part'] = 2.7
M.SHAPES['big']['head'] = 4.4
M.SHAPES['big']['chest'] = 4.2
M.SHAPES['mrrrp']['LIFT_U'] = 2.0
M.SHAPES['mrrrp']['DROP_U'] = 4.0
M.SHAPES['growl']['jaw_down'] = 1.6

# Ensure BOX exists for the 1–2 lead-in frames.
for _src, _dsts in ((40, (38, 39)), (96, (94, 95)), (112, (113, 114, 115, 116, 117))):
    if _src in M.BOX:
        for _d in _dsts:
            if _d not in M.BOX:
                M.BOX[_d] = list(M.BOX[_src])


# ---------- review-item envelopes (isolated / placed stem, not mix RMS) ----------
# meow1 stem: ph 40–56, peak 49. Open starts 2 frames before onset.
# meow2 stem: ph 96–112, peak 102.
def _env(ph, start, peak, end):
    if ph < start or ph > end:
        return 0.0, 1.0
    if ph <= peak:
        p = float(sstep((ph - start) / max(peak - start, 1e-3)))
        return p, 0.0
    p = float(1.0 - sstep((ph - peak) / max(end - peak, 1e-3)))
    cl = float(sstep((ph - peak) / max(end - peak, 1e-3)))
    return p, cl

def p_geom_ph(ph, start, peak, end):
    ks = (-2, -1, 0, 1, 2)
    ws = (1.0, 2.0, 3.0, 2.0, 1.0)
    ap = ac = sw = 0.0
    for k, w in zip(ks, ws):
        p, cl = _env(ph + k, start, peak, end)
        ap += w * p
        ac += w * cl
        sw += w
    return ap / sw, ac / sw

def style_and_p(i):
    """Return (style, p, closing, tg) or (None, 0, 1, None)."""
    ph = phase_of(i)
    # medium: first meow-1 in the 25s span
    if 125 <= i <= 160:
        p, cl = p_geom_ph(ph, 38.0, 49.0, 60.0)
        if p <= 1e-4:
            return None, 0.0, 1.0, None
        tg = 10.00 + 0.50 * p
        return 'medium', p, cl, tg
    # mrrrp
    if 178 <= i <= 220:
        p, cl = p_geom_ph(ph, 94.0, 102.0, 124.0)
        if p <= 1e-4:
            return None, 0.0, 1.0, None
        return 'mrrrp', p, cl, None
    # growl (second meow-1 window): sealed
    if 360 <= i <= 400:
        p, cl = p_geom_ph(ph, 38.0, 49.0, 60.0)
        if p <= 1e-4:
            return None, 0.0, 1.0, None
        return 'growl', p, cl, None
    # big
    if 418 <= i <= 460:
        p, cl = p_geom_ph(ph, 94.0, 102.0, 118.0)
        if p <= 1e-4:
            return None, 0.0, 1.0, None
        tg = 10.00 + 0.55 * p
        return 'big', p, cl, tg
    return None, 0.0, 1.0, None


# ---------- laptop screen (241-periodic character animation) ----------
# Perspective screen interior (same geometry as v41b).
def screen_mask():
    y_s = np.clip((Y - 378.0) / (556.0 - 378.0), 0, 1)
    x_right = 150.0 + 48.0 * y_s
    screen = (sstep((X - 8) / 6) * sstep((x_right - X) / 7)
              * sstep((Y - 378) / 7) * sstep((556 - Y) / 7))
    return cv2.GaussianBlur(np.clip(screen, 0, 1), (0, 0), 2.0)

# Slightly inset corners so we stay inside the live cartoon, not the bezel.
SCREEN_SRC = np.array([[14, 384], [146, 384], [192, 550], [14, 550]], np.float32)
SW, SH = 400, 512
SCREEN_DST = np.array([[0, 0], [SW - 1, 0], [SW - 1, SH - 1], [0, SH - 1]], np.float32)
H_FWD = cv2.getPerspectiveTransform(SCREEN_SRC, SCREEN_DST)
H_INV = cv2.getPerspectiveTransform(SCREEN_DST, SCREEN_SRC)

def unwarp_screen(frame):
    return cv2.warpPerspective(frame, H_FWD, (SW, SH), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def warp_screen(rect):
    return cv2.warpPerspective(rect, H_INV, (W, H), flags=cv2.INTER_LINEAR)

def _soft_disk(h, w, cx, cy, rx, ry):
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    m = 1.0 - np.sqrt(((xx - cx) / max(rx, 1)) ** 2 + ((yy - cy) / max(ry, 1)) ** 2)
    return cv2.GaussianBlur(np.clip(m, 0, 1), (0, 0), 2.2)


def build_laptop_layers(ref_frame):
    """Tight feathered character layers + inpainted background from one still."""
    rect = unwarp_screen(ref_frame)
    hsv = cv2.cvtColor(rect, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    b, g, r = cv2.split(rect.astype(np.float32))
    # Teal / dusty wall
    wall = sstep(((h - 85) / 8.0) * ((110 - h) / 8.0) + 0.5)  # rough
    wall = sstep((h - 80) / 8.0) * sstep((115 - h) / 10.0) * sstep((s - 30) / 20.0)
    # Yellow door / frame (static)
    door = sstep((h - 12) / 6.0) * sstep((38 - h) / 8.0) * sstep((s - 40) / 20.0)
    # Confine the door prior to the upper-middle rectangle
    yy, xx = np.mgrid[:SH, :SW].astype(np.float32)
    door_geo = sstep((xx - 0.38 * SW) / 8) * sstep((0.78 * SW - xx) / 8) * sstep((yy - 0.08 * SH) / 8) * sstep((0.55 * SH - yy) / 10)
    door = door * door_geo
    static = cv2.GaussianBlur(np.clip(np.maximum(wall, door * 0.85), 0, 1), (0, 0), 1.6)

    def disk(cx, cy, rx, ry):
        m = 1.0 - np.sqrt(((xx - cx) / max(rx, 1)) ** 2 + ((yy - cy) / max(ry, 1)) ** 2)
        return cv2.GaussianBlur(np.clip(m, 0, 1), (0, 0), 2.8)

    bugs = disk(0.30 * SW, 0.56 * SH, 0.32 * SW, 0.42 * SH)
    mid  = disk(0.58 * SW, 0.50 * SH, 0.28 * SW, 0.40 * SH)
    right= disk(0.80 * SW, 0.64 * SH, 0.22 * SW, 0.32 * SH)
    # Tighten: keep high-contrast / non-static pixels inside each prior.
    # Character colours: black, white, salmon, green, saturated blue (not wall teal).
    black = sstep((50 - v) / 20.0)
    white = sstep((v - 170) / 25.0) * sstep((40 - s) / 20.0)
    salmon = sstep((h - 0) / 4) * sstep((25 - h) / 6) * sstep((s - 50) / 20) * sstep((v - 60) / 20)
    green = sstep((h - 35) / 8) * sstep((85 - h) / 8) * sstep((s - 40) / 20)
    blue = sstep((h - 95) / 8) * sstep((140 - h) / 10) * sstep((s - 50) / 20) * sstep((v - 50) / 20)
    charcol = np.clip(np.maximum.reduce([black, white, salmon, green, blue]), 0, 1)
    charcol = cv2.GaussianBlur(charcol, (0, 0), 1.4)
    bugs = np.clip(bugs * (0.15 + 0.95 * charcol) * (1.0 - 0.90 * static), 0, 1)
    mid  = np.clip(mid  * (0.18 + 0.95 * charcol) * (1.0 - 0.80 * static), 0, 1)
    right= np.clip(right* (0.18 + 0.95 * charcol) * (1.0 - 0.90 * static), 0, 1)
    # Clean overlaps: Bugs in front on the left, right char in front on the right.
    mid = np.clip(mid * (1.0 - 0.70 * bugs), 0, 1)
    right = np.clip(right * (1.0 - 0.55 * mid) * (1.0 - 0.40 * bugs), 0, 1)
    # Harden cores, feather edges
    def firm(m):
        m = np.clip(m, 0, 1)
        core = cv2.erode((m > 0.35).astype(np.uint8),
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))).astype(np.float32)
        m = np.maximum(m * 0.55, core)
        return cv2.GaussianBlur(np.clip(m, 0, 1), (0, 0), 1.8)
    bugs, mid, right = firm(bugs), firm(mid), firm(right)
    union = np.clip(np.maximum.reduce([bugs, mid, right]), 0, 1)
    # Inpaint characters out of the background once.
    miss = (union > 0.45).astype(np.uint8)
    if int(miss.sum()) > 0:
        bg = cv2.inpaint(rect, miss, 6, cv2.INPAINT_TELEA).astype(np.float32)
    else:
        bg = rect.astype(np.float32)
    # Softly mix so leftover mask edges don't show a cut-out.
    a = cv2.GaussianBlur(union, (0, 0), 2.2)
    blur = cv2.GaussianBlur(rect.astype(np.float32), (0, 0), 6.0)
    bg = rect.astype(np.float32) * (1.0 - a[..., None]) + bg * a[..., None]
    bg = bg * (1.0 - 0.25 * a[..., None]) + blur * (0.25 * a[..., None])
    return {
        'bg': bg,
        'bugs': (rect.astype(np.float32), bugs),
        'mid': (rect.astype(np.float32), mid),
        'right': (rect.astype(np.float32), right),
        'union': union,
        'rect0': rect,
    }

def shift_rect(img, mask, dx, dy):
    Maf = np.array([[1, 0, dx], [0, 1, dy]], np.float32)
    im = cv2.warpAffine(img, Maf, (SW, SH), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    ma = cv2.warpAffine(mask, Maf, (SW, SH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return im, np.clip(ma, 0, 1)

def animate_laptop(frame, layers, ph, screen):
    t = 2.0 * np.pi * ph / NCYCLE
    # Visible, gentle, integer-cycle bobs. No extra double-paste gesture layers.
    def offs(k, ax, ay, phi):
        return (ax * np.sin(k * t + phi), ay * np.sin(k * t + phi + 0.7))
    bx, by = offs(2.0, 2.8, 6.0, 0.2)
    mx, my = offs(2.0, 2.4, 5.2, 2.1)
    rx, ry = offs(2.0, 2.2, 5.5, 4.0)
    mx += 1.0 * np.sin(t + 0.5)
    my += 0.8 * np.sin(t + 1.1)

    comp = layers['bg'].copy()
    for name, dx, dy in (('mid', mx, my), ('right', rx, ry), ('bugs', bx, by)):
        im, ma = shift_rect(*layers[name], dx, dy)
        a = ma[..., None]
        comp = comp * (1.0 - a) + im * a

    glow = 1.0 + 0.022 * np.sin(2.0 * t + 0.3) + 0.010 * np.sin(3.0 * t)
    warm = np.array([0.985, 1.00, 1.025], np.float32)
    comp = np.clip(comp * glow * warm, 0, 255)

    back = cv2.warpPerspective(
        np.clip(comp, 0, 255).astype(np.uint8), H_INV, (W, H),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
    ).astype(np.float32)
    a = screen[..., None]
    out = frame.astype(np.float32) * (1.0 - a) + back * a
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------- ear rigidity + halo decontamination ----------
def mullion_mask():
    mullion = (sstep((X - 615) / 4) * sstep((658 - X) / 4)
               * sstep((Y - 70) / 8) * sstep((420 - Y) / 8))
    mullion = np.maximum(mullion, sstep((X - 220) / 6) * sstep((250 - X) / 6)
                         * sstep((Y - 70) / 8) * sstep((420 - Y) / 8))
    mullion = np.maximum(mullion, sstep((X - 425) / 5) * sstep((465 - X) / 5)
                         * sstep((Y - 70) / 8) * sstep((420 - Y) / 8))
    mullion = np.maximum(mullion, sstep((X - 855) / 5) * sstep((915 - X) / 5)
                         * sstep((Y - 70) / 8) * sstep((420 - Y) / 8))
    return cv2.GaussianBlur(np.clip(mullion, 0, 1), (0, 0), 1.2)

def left_ear_lock(cur, template, i, Title, Wglass, Steam, Mouth, Eyes, Mull):
    """Hold the viewer-left ear silhouette rigid through the 15.4s head turn."""
    # Feathered weight: full lock n368–378, ease 360–366 and 380–388.
    if i < 360 or i > 388:
        return cur
    wt = 1.0
    if i < 368:
        wt = float(sstep((i - 360) / 8.0))
    elif i > 378:
        wt = float(1.0 - sstep((i - 378) / 10.0))
    if wt < 0.02:
        return cur
    box = C.ear_rois()['L']
    dx, dy, resp = C.align_roi(template, cur, box)
    aligned = C.shift_img(template.astype(np.float32), dx, dy)
    fur_t = C.shift_img(C.ear_color_mask(template), dx, dy)
    fur_c = C.ear_color_mask(cur)
    # Ear only: template fur, not post / glass / title / mouth / eyes.
    protect = np.clip(np.maximum.reduce([Title, Steam, Mouth, Eyes, Mull]), 0, 1)
    # Keep snow: don't overwrite glass unless it's fur.
    glass = Wglass * (1.0 - sstep((fur_t - 0.25) / 0.25))
    a = np.clip(fur_t, 0, 1) * (1.0 - protect) * (1.0 - glass)
    y0, y1, x0, x1 = box
    boxm = np.zeros((H, W), np.float32)
    boxm[y0:y1, x0:x1] = 1.0
    boxm = cv2.GaussianBlur(boxm, (0, 0), 6.0)
    a = a * boxm * wt
    # Slightly prefer the pointed template silhouette at the tip (upper half of box).
    tip = np.zeros((H, W), np.float32)
    tip[y0:y0 + (y1 - y0) // 2, x0:x1] = 1.0
    tip = cv2.GaussianBlur(tip, (0, 0), 6.0)
    a = np.clip(a * (0.75 + 0.35 * tip), 0, 1)
    a = cv2.GaussianBlur(a, (0, 0), 0.8)
    out = cur.astype(np.float32) * (1.0 - a[..., None]) + aligned * a[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)

def decontaminate_ear_halo(cur, Mull, Wglass, Title, Steam, Mouth, Eyes):
    """Edge-aware matte decontamination where the viewer-right ear meets the post.
    No blur of the fur. Unmix light fringe toward post or fur.
    """
    fur = C.ear_color_mask(cur)
    y0, y1, x0, x1 = C.ear_rois()['R']
    boxm = np.zeros((H, W), np.float32)
    boxm[y0:y1, x0:x1] = 1.0
    boxm = cv2.GaussianBlur(boxm, (0, 0), 4.0)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    hard = (fur > 0.35).astype(np.uint8)
    dil = cv2.dilate(hard, ker).astype(np.float32)
    ero = cv2.erode(hard, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))).astype(np.float32)
    ring = np.clip(dil - ero, 0, 1)
    # Only the side that touches the dark post (right of the right ear).
    near_post = sstep((X - 820) / 12) * sstep((930 - X) / 10) * sstep((Y - 130) / 10) * sstep((300 - Y) / 12)
    ring = ring * near_post * boxm
    ring = cv2.GaussianBlur(ring, (0, 0), 0.6)
    if float(ring.max()) < 1e-4:
        return cur
    # Sample post and fur colours.
    post_w = Mull * sstep((X - 855) / 4) * sstep((915 - X) / 4) * (1.0 - sstep((fur - 0.25) / 0.2))
    fur_w = fur * boxm * (1.0 - Mull) * ero
    img = cur.astype(np.float32)
    def wmean(mask):
        s = float(mask.sum())
        if s < 8:
            return None
        return (img * mask[..., None]).sum(axis=(0, 1)) / s
    Pcol = wmean(post_w)
    Fcol = wmean(fur_w)
    if Pcol is None or Fcol is None:
        return cur
    # Unmix each fringe pixel: C ≈ a F + (1-a) P, a in [0,1].
    d = Fcol - Pcol
    n2 = float(np.dot(d, d)) + 1e-6
    Cpix = img
    a_est = np.sum((Cpix - Pcol) * d, axis=2) / n2
    a_est = np.clip(a_est, 0.0, 1.0)
    # Decontaminated fur colour
    dec = (Cpix - (1.0 - a_est)[..., None] * Pcol) / np.maximum(a_est[..., None], 0.15)
    dec = np.clip(dec, 0, 255)
    # Snap: post-side fringe (low a) goes to post; fur-side to decontaminated fur.
    # This kills the light halo without blurring the tuft.
    to_post = sstep((0.48 - a_est) / 0.25)
    to_fur = 1.0 - to_post
    target = Pcol * to_post[..., None] + dec * to_fur[..., None]
    w = np.clip(ring * (1.0 - Title) * (1.0 - Steam) * (1.0 - Mouth) * (1.0 - Eyes), 0, 1)
    # Slightly stronger where luma is lighter than both (classic halo).
    luma = 0.114 * img[..., 0] + 0.587 * img[..., 1] + 0.299 * img[..., 2]
    lp = 0.114 * Pcol[0] + 0.587 * Pcol[1] + 0.299 * Pcol[2]
    lf = 0.114 * Fcol[0] + 0.587 * Fcol[1] + 0.299 * Fcol[2]
    halo = sstep((luma - np.maximum(lp, lf) + 4) / 10.0)
    w = np.clip(w * (0.70 + 0.50 * halo), 0, 1)
    out = img * (1.0 - w[..., None]) + target * w[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------- growl sealed mouth line ----------
def _crease_band(box, lift=0.0):
    x0, y0, w, h = box
    # Use the canonical mouth patch geometry.
    yy, xx = np.mgrid[0:M.PH, 0:M.PW].astype(np.float32)
    dxx = xx - 48
    dyy = yy - 20 - lift
    band = np.exp(-0.5 * (dyy / 3.2) ** 2) * (1.0 - sstep((np.abs(dxx) - 20.0) / 7.0))
    band = cv2.GaussianBlur(band.astype(np.float32), (0, 0), 0.6)
    return band, (x0, y0)

def stamp_growl_line(cur, donor, ph, p, i):
    """Crisp sealed crease, rigidly translated with the snout."""
    if ph not in M.BOX:
        return cur
    box = list(M.BOX[ph])
    x0, y0, w, h = box[0], box[1], M.PW, M.PH
    # Align donor (closed, sharp) to current snout.
    abox = (max(0, y0 - 20), min(H, y0 + h + 20), max(0, x0 - 20), min(W, x0 + w + 20))
    dx, dy, resp = C.align_roi(donor, cur, abox)
    aligned = C.shift_img(donor.astype(np.float32), dx, dy)
    band, _ = _crease_band(box, lift=0.0)
    # Sharpen the donor crease slightly (unsharp on the band only).
    crop_a = aligned[y0:y0 + M.PH, x0:x0 + M.PW]
    crop_c = cur[y0:y0 + M.PH, x0:x0 + M.PW].astype(np.float32)
    if crop_a.shape[:2] != (M.PH, M.PW) or crop_c.shape[:2] != (M.PH, M.PW):
        return cur
    blur = cv2.GaussianBlur(crop_a, (0, 0), 0.8)
    sharp = np.clip(crop_a + 0.85 * (crop_a - blur), 0, 255)
    # Darken the crease just a touch so it stays readable, still fur-coloured.
    luma = 0.114 * sharp[..., 0] + 0.587 * sharp[..., 1] + 0.299 * sharp[..., 2]
    # pull mid-luma crease slightly darker, not a black ink line
    darken = sstep((140 - luma) / 50.0) * sstep((luma - 40) / 20.0)
    sharp = sharp * (1.0 - 0.18 * darken * band)[..., None]
    w = np.clip(band * (0.55 + 0.45 * p), 0, 1)[..., None]
    pasted = crop_c * (1.0 - w) + sharp * w
    out = cur.copy()
    out[y0:y0 + M.PH, x0:x0 + M.PW] = np.clip(np.round(pasted), 0, 255).astype(np.uint8)
    return out


# ---------- open-mouth paint + full-frame jaw ----------

def restyle_cavity(frame, ph, p, style):
    """Smooth dark-to-warm depth plus a soft pink tongue mound (no stripe bands)."""
    if style not in ('medium', 'big') or p < 0.18 or ph not in M.BOX:
        return frame
    x0, y0, _, _ = M.BOX[ph]
    y0, x0 = int(y0), int(x0)
    y1, x1 = y0 + M.PH, x0 + M.PW
    if y1 > H or x1 > W or y0 < 0 or x0 < 0:
        return frame
    roi = frame[y0:y1, x0:x1].astype(np.float32)
    b, g, r = roi[..., 0], roi[..., 1], roi[..., 2]
    luma = 0.114 * b + 0.587 * g + 0.299 * r
    dark = sstep((125.0 - luma) / 35.0)
    warm = sstep(((r - g) + 8.0) / 16.0)
    inside = np.clip(dark * (0.35 + 0.65 * warm), 0, 1)
    inside = cv2.GaussianBlur(inside, (0, 0), 0.7)
    if float(inside.max()) < 0.2:
        return frame
    yy, xx = np.mgrid[0:M.PH, 0:M.PW].astype(np.float32)
    mid = M.MOUTH_MID
    col_on = inside > 0.28
    yU = np.full(M.PW, np.nan, np.float32)
    yL = np.full(M.PW, np.nan, np.float32)
    for x in range(M.PW):
        ys = np.where(col_on[:, x])[0]
        if len(ys) >= 2:
            yU[x] = float(ys[0])
            yL[x] = float(ys[-1])
    xs = np.arange(M.PW, dtype=np.float32)
    good = np.isfinite(yU)
    if good.sum() < 8:
        return frame
    yU = np.interp(xs, xs[good], yU[good])
    yL = np.interp(xs, xs[good], yL[good])
    yU = cv2.GaussianBlur(yU.reshape(1, -1), (0, 0), 2.0).ravel()
    yL = cv2.GaussianBlur(yL.reshape(1, -1), (0, 0), 2.0).ravel()
    gap = np.maximum(yL - yU, 1e-3)
    f = np.clip((yy - yU[None, :]) / gap[None, :], 0, 1)
    au = np.abs(xx - mid) / 28.0
    cav = inside * (1.0 - sstep((au - 1.05) / 0.2))
    fang = sstep((luma - 150) / 25.0) * sstep((0.42 - f) / 0.15) * cav
    fang = cv2.GaussianBlur(fang, (0, 0), 0.5)
    # Smooth cavity: darkest at the back/top, warming downward. No hard stops.
    BACK = np.array([24.0, 34.0, 70.0], np.float32)
    WARM = np.array([52.0, 64.0, 118.0], np.float32)
    depth = np.clip(0.15 + 0.85 * f, 0, 1)  # 0 at teeth/top, 1 near lower lip
    depth = depth * (0.85 + 0.15 * (1.0 - np.clip(au, 0, 1)))
    tex = BACK * (1.0 - depth)[..., None] + WARM * depth[..., None]
    # Tongue: one soft elliptical mound in the lower mouth, rounded shading.
    # Centre sits low; a broad, low-contrast highlight on the upper-front (lamp-left).
    mx = mid - 3.0
    fy = (f - 0.68) / 0.20
    fx = (xx - mx) / 13.0
    rr = fx * fx + fy * fy
    mound = np.exp(-0.5 * rr / 0.85) * sstep((f - 0.34) / 0.22) * (1.0 - sstep((f - 0.94) / 0.08))
    mound = mound * (1.0 - sstep((au - 0.85) / 0.18))
    # Height field for shading (0 at rim, 1 at crest)
    height = np.clip(1.0 - rr / 1.15, 0, 1) * mound
    # Soft lighting: slight darkening toward the root (top of mound / throat)
    # and a wide highlight slightly above and left of centre — no thin stripe.
    TONG = np.array([108.0, 120.0, 198.0], np.float32)
    TONG_LIT = np.array([142.0, 150.0, 216.0], np.float32)
    TONG_SH = np.array([72.0, 80.0, 142.0], np.float32)
    hl = np.exp(-0.5 * (((f - 0.60) / 0.14) ** 2 + ((xx - (mid - 5.0)) / 9.0) ** 2))
    hl = hl * mound
    sh = (1.0 - height) * mound
    tcol = TONG * (1.0 - 0.50 * sh)[..., None] + TONG_SH * (0.50 * sh)[..., None]
    tcol = tcol * (1.0 - 0.48 * hl)[..., None] + TONG_LIT * (0.48 * hl)[..., None]
    tex = tex * (1.0 - 0.82 * mound)[..., None] + tcol * (0.82 * mound)[..., None]
    # Soft lower-lip contact shadow (wide, not a line).
    lip_sh = sstep((f - 0.86) / 0.10) * (1.0 - sstep((f - 0.99) / 0.04)) * (1.0 - sstep((au - 0.85) / 0.15))
    shc = np.array([42.0, 54.0, 90.0], np.float32)
    tex = tex * (1.0 - 0.35 * lip_sh)[..., None] + shc * (0.35 * lip_sh)[..., None]
    w = cav * (1.0 - fang) * sstep((p - 0.18) / 0.15)
    w = cv2.GaussianBlur(w, (0, 0), 0.8)
    out_roi = roi * (1.0 - w[..., None]) + tex * w[..., None]
    out_roi = out_roi * (1.0 - fang[..., None]) + roi * fang[..., None]
    out = frame.copy()
    out[y0:y1, x0:x1] = np.clip(np.round(out_roi), 0, 255).astype(np.uint8)
    return out


def jaw_drop(frame, ph, p, style, box_offset=(0, 0)):
    """Rigid feathered chin/jaw translate, full-frame, outside the in-box remap."""
    if p < 0.05 or ph not in M.BOX:
        return frame
    amp = {'medium': 4.6, 'big': 6.0, 'mrrrp': 1.7, 'growl': 1.2}.get(style, 0.0) * float(sstep(p))
    if amp < 0.15:
        return frame
    x0, y0, _, _ = M.BOX[ph]
    x0 += box_offset[0]
    y0 += box_offset[1]
    # Start just below the lip line (box y ≈ 20 + drop), fade before the chest.
    lip_y = y0 + 28.0
    mx = x0 + M.MOUTH_MID
    m = sstep((Y - (lip_y + 2)) / 7.0)
    m = m * (1.0 - sstep((Y - (lip_y + 62)) / 14.0))
    rad = 1.0 - np.sqrt(((X - mx) / 78.0) ** 2 + ((Y - (lip_y + 22)) / 48.0) ** 2)
    m = m * sstep(rad)
    # Do not move eyes / title / mouth cavity (opening stays, chin follows).
    # Allow the lower lip and chin. Zero at and above mid-mouth.
    m = m * (1.0 - sstep((lip_y - 6 - Y) / 6.0))
    m = cv2.GaussianBlur(np.clip(m, 0, 1), (0, 0), 3.0)
    shifted = C.shift_img(frame.astype(np.float32), 0.0, amp)
    out = frame.astype(np.float32) * (1.0 - m[..., None]) + shifted * m[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)

def paint_open_mouth(cur, donor, ph, p, closing, tg, style):
    if ph not in M.BOX:
        return cur
    M.STYLE = style
    M.T_NOW = ph / 24.0
    # Align closed donor to the current snout so we paint on a closed base
    # that already has v41c lighting.
    x0, y0, _, _ = M.BOX[ph]
    abox = (max(0, y0 - 30), min(H, y0 + M.PH + 50), max(0, x0 - 30), min(W, x0 + M.PW + 30))
    dx, dy, resp = C.align_roi(donor, cur, abox)
    aligned = C.shift_img(donor, dx, dy)
    # Paint in current coordinates: shift BOX by the residual if needed.
    # aligned donor mouth should now sit on the current snout; BOX is cycle-based.
    # Measure a residual using the same box on aligned vs cycle expectation — skip,
    # align_roi already moved donor onto `cur`.
    try:
        patch = M.render_one(ph, p, closing, tg if tg is not None else 10.2, aligned, y_off=0)
    except Exception as e:
        print('render_one failed', ph, style, e, flush=True)
        return cur
    # Soft oval paste of the painted patch.
    yy, xx = np.mgrid[0:M.PH, 0:M.PW].astype(np.float32)
    dxx = xx - M.MOUTH_MID
    dyy = yy - 20
    oval = np.sqrt((dxx / 30.0) ** 2 + ((dyy - 6) / 22.0) ** 2)
    om = 1.0 - sstep((oval - 0.92) / 0.40)
    om = cv2.GaussianBlur(om.astype(np.float32), (0, 0), 1.2)
    y0i, x0i = int(y0), int(x0)
    if y0i + M.PH > H or x0i + M.PW > W or y0i < 0 or x0i < 0:
        return cur
    crop = cur[y0i:y0i + M.PH, x0i:x0i + M.PW].astype(np.float32)
    if crop.shape[:2] != patch.shape[:2]:
        return cur
    pasted = crop * (1.0 - om[..., None]) + patch.astype(np.float32) * om[..., None]
    out = cur.copy()
    out[y0i:y0i + M.PH, x0i:x0i + M.PW] = np.clip(np.round(pasted), 0, 255).astype(np.uint8)
    out = restyle_cavity(out, ph, p, style)
    return out


# ---------- encode ----------
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


def process(src, preview=None):
    Title = C.title_mask()
    Wglass = C.window_glass_mask()
    Steam = C.steam_mask()
    Mouth = C.mouth_mask()
    Eyes = C.eye_mask()
    Mull = mullion_mask()
    screen = screen_mask()

    layers = build_laptop_layers(src[0])
    # Closed-mouth donors (v41c, same lighting), taken just before each event.
    donor_medium = src[126]
    donor_mrrrp = src[178]
    donor_growl = src[363]
    donor_big = src[422]
    ear_template = src[368]  # pointed left-ear pose just before the reported flatten

    want = set(preview) if preview else set(range(NFR))
    out = [None] * NFR
    for i in range(NFR):
        if i not in want:
            continue
        comp = src[i].copy()
        ph = phase_of(i)

        # 1. Laptop animation (bezel/body stay from src; only the screen interior).
        comp = animate_laptop(comp, layers, ph, screen)

        # 3. Right-ear / post halo decontamination (all frames, cheap).
        comp = decontaminate_ear_halo(comp, Mull, Wglass, Title, Steam, Mouth, Eyes)

        # 2. Left-ear rigidity through the 15.4–15.6s turn.
        comp = left_ear_lock(comp, ear_template, i, Title, Wglass, Steam, Mouth, Eyes, Mull)

        # 6/7. Mouth + jaw.
        st, p, cl, tg = style_and_p(i)
        if st is not None:
            if st == 'medium':
                comp = paint_open_mouth(comp, donor_medium, ph, p, cl, tg, st)
            elif st == 'big':
                comp = paint_open_mouth(comp, donor_big, ph, p, cl, tg, st)
            elif st == 'mrrrp':
                # Keep a small opening if the existing painter produces one; otherwise
                # just ease a tiny jaw. Use the closed donor + tiny p.
                if p > 0.35:
                    comp = paint_open_mouth(comp, donor_mrrrp, ph, min(p, 0.55), cl, 10.1, st)
            elif st == 'growl':
                comp = stamp_growl_line(comp, donor_growl, ph, p, i)
            comp = jaw_drop(comp, ph, p, st)

        # Title is the approved scrolling card — never retouch.
        comp[14:60, 16:165] = src[i][14:60, 16:165]
        out[i] = comp
        if i % 25 == 0 or i in (133, 138, 191, 374, 430, 432):
            print(f'  frame {i} ph={ph} style={st} p={p:.3f}', flush=True)
    return out


def save_preview(src, out, idx):
    q = QC / 'preview'
    for i in idx:
        if out[i] is None:
            continue
        cv2.imwrite(str(q / f'full-n{i}.jpg'), out[i], [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        # mouth
        a = src[i][350:470, 600:780]
        b = out[i][350:470, 600:780]
        cv2.imwrite(str(q / f'mouth-n{i}.jpg'), np.hstack([a, b]), [int(cv2.IMWRITE_JPEG_QUALITY), 93])
        # jaw
        a = src[i][360:560, 580:820]
        b = out[i][360:560, 580:820]
        cv2.imwrite(str(q / f'jaw-n{i}.jpg'), np.hstack([a, b]), [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        # ears
        a = src[i][140:310, 480:940]
        b = out[i][140:310, 480:940]
        cv2.imwrite(str(q / f'ears-n{i}.jpg'), np.hstack([a, b]), [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        # laptop
        a = src[i][370:570, 0:220]
        b = out[i][370:570, 0:220]
        cv2.imwrite(str(q / f'laptop-n{i}.jpg'), np.hstack([a, b]), [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    # laptop grid of new animation
    tiles = []
    for i in (0, 30, 60, 90, 120, 150, 180, 210, 240):
        if i < NFR and out[i] is not None:
            t = out[i][378:556, 0:200].copy()
            cv2.putText(t, f'n{i}', (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.putText(t, f'n{i}', (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            tiles.append(t)
    if len(tiles) >= 9:
        grid = np.vstack([np.hstack(tiles[0:3]), np.hstack(tiles[3:6]), np.hstack(tiles[6:9])])
        cv2.imwrite(str(q / 'laptop-anim-grid.jpg'), grid, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    print('preview written', q)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--preview', action='store_true')
    args = ap.parse_args()

    for label, path, expect in (
        ('v41', V41_MP4, V41_SHA),
        ('v41b-r2', V41B_R2, R2_SHA),
        ('v41c', V41C_MP4, V41C_SHA),
        ('pcm', PCM, PCM_SHA),
    ):
        got = sha256(path)
        if got != expect:
            raise SystemExit(f'{label} sha changed: {got}')
    print('frozen shas ok', flush=True)
    print('loading v41c lossless…', flush=True)
    src = load_span(SRC)

    if args.preview:
        idx = [0, 30, 60, 90, 120, 150, 180, 210, 240,
               126, 127, 129, 133, 138, 145, 155,
               185, 191, 210,
               366, 368, 370, 372, 374, 378, 384,
               424, 430, 432, 440]
        out = process(src, preview=idx)
        save_preview(src, out, idx)
        return

    out = process(src)
    # title identity vs source
    bad = [i for i, (a, b) in enumerate(zip(src, out)) if not np.array_equal(a[14:60, 16:165], b[14:60, 16:165])]
    print('title mismatches' if bad else 'title core byte-identical', len(bad) if bad else '')
    los = WORK / 'v41d-span-890-915-lossless.mkv'
    mp4 = P / 'v41d-test-25s.mp4'
    print('writing lossless…', flush=True)
    write_lossless(out, los)
    print('encoding mp4…', flush=True)
    encode_mp4(out, mp4)
    print('v41d mp4', sha256(mp4))
    print('v41 still', sha256(V41_MP4))
    print('r2 still', sha256(V41B_R2))
    print('v41c still', sha256(V41C_MP4))

if __name__ == '__main__':
    main()
