#!/usr/bin/env python3
"""v41b: parallax, static prop plates, snappier blinks, static lamp lighting.

Base is the approved v41 r6 25s span (lossless). v41-test-25s.mp4 is never written.
Protected (copied back from the v41 source every frame):
  scrolling title, window/snow, steam, ears.
Audio PCM is identical to v41.
"""
from __future__ import annotations
import json, hashlib, subprocess as sp
from pathlib import Path
import cv2
import numpy as np

P = Path('/workspace/lofi-v41-meow')
QC = P / 'qc' / 'v41b'
QC.mkdir(parents=True, exist_ok=True)
WORK = P / 'v41b'
WORK.mkdir(exist_ok=True)

SRC_LOSSLESS = P / 'work' / 'v41-span-890-915-lossless.mkv'
SRC_MP4 = P / 'v41-test-25s.mp4'          # read-only
PCM = P / 'work' / 'test-audio-890-915-v41.f32le'
V41_SHA = 'f53cb1127c75775b740b555dd856e0002c784f80a2a3097118d8e197a794e95f'
NFR, W, H, FPS = 600, 1280, 720, 24

def sstep(x):
    x = np.clip(x, 0, 1)
    return x * x * x * (x * (6 * x - 15) + 10)

def smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)

def sha256(path, limit=None):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        if limit is None:
            for c in iter(lambda: f.read(1 << 20), b''):
                h.update(c)
        else:
            h.update(f.read(limit))
    return h.hexdigest()


def load_src():
    cap = cv2.VideoCapture(str(SRC_LOSSLESS))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    assert len(frames) == NFR, len(frames)
    return frames


# --------------- masks ---------------
# Window glass (snow lives here). Built from the official safe-window-mask,
# expanded to the full pane, then with the cat's head / ears cut out per frame.
def window_glass_mask():
    wm = cv2.imread('/workspace/lofi-v30/safe-window-mask.png', 0).astype(np.float32) / 255.0
    # Completing the pane: the official mask misses the right-hand glass and
    # a strip of the lower right. A soft rectangle covers the glass only.
    Y, X = np.mgrid[:H, :W].astype(np.float32)
    # glass rectangle (inside the wooden frame)
    pane = sstep((X - 228) / 10) * sstep((640 - X) / 10) * sstep((Y - 78) / 8) * sstep((405 - Y) / 8)
    # don't include the wooden mullion strongly — keep it in "window unchanged"
    # by using the union: both glass and official mask
    m = np.maximum(wm, pane * 0.95)
    m = cv2.GaussianBlur(m, (0, 0), 1.6)
    return np.clip(m, 0, 1)

def steam_mask():
    sm = cv2.imread('/workspace/lofi-v28/steam-mask.png', 0).astype(np.float32) / 255.0
    M = np.zeros((H, W), np.float32)
    y0, x0 = 389, 394
    M[y0:y0 + sm.shape[0], x0:x0 + sm.shape[1]] = sm
    M = cv2.GaussianBlur(M, (0, 0), 2.4)
    return np.clip(M, 0, 1)

Y_GRID, X_GRID = np.mgrid[:H, :W].astype(np.float32)

def title_mask():
    m = np.zeros((H, W), np.float32)
    m[14:60, 16:165] = 1.0
    m = cv2.GaussianBlur(m, (0, 0), 1.2)
    return np.clip(m, 0, 1)


def cat_prior():
    """Soft spatial prior of where Mimi usually sits (head, chest, paws)."""
    # body ellipse
    body = sstep(1.0 - np.sqrt(((X_GRID - 710) / 260) ** 2 + ((Y_GRID - 430) / 230) ** 2))
    # paws on the desk
    paw = sstep(1.0 - np.sqrt(((X_GRID - 620) / 120) ** 2 + ((Y_GRID - 560) / 70) ** 2))
    return np.clip(np.maximum(body, paw), 0, 1)


def cat_mask_frame(f, prior):
    """Per-frame fur mask so plates never cover the cat."""
    b, g, r = cv2.split(f.astype(np.float32))
    luma = 0.114 * b + 0.587 * g + 0.299 * r
    # cream / white fur
    white = sstep((luma - 145) / 25) * sstep((r - b + 10) / 20) * sstep((40 - np.abs(r - g)) / 20)
    # orange fur
    orange = sstep((r - 90) / 25) * sstep((r - g - 15) / 15) * sstep((r - b - 30) / 20) * sstep((200 - luma) / 40)
    fur = np.maximum(white, orange)
    fur = fur * (0.25 + 0.75 * prior)
    # drop obvious props: very dark (mic/headphones) and saturated green (plants)
    hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV).astype(np.float32)
    green = sstep((hsv[..., 0] - 35) / 8) * sstep((80 - hsv[..., 0]) / 8) * sstep((hsv[..., 1] - 50) / 30)
    dark = sstep((70 - luma) / 20)
    fur = fur * (1.0 - 0.9 * green) * (1.0 - 0.85 * dark)
    fur = cv2.GaussianBlur(fur, (0, 0), 2.2)
    # fill small holes inside the body
    bin = (fur > 0.35).astype(np.uint8) * 255
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    bin = cv2.morphologyEx(bin, cv2.MORPH_CLOSE, ker)
    fur = np.maximum(fur, cv2.GaussianBlur(bin.astype(np.float32) / 255.0, (0, 0), 2.0) * 0.7)
    return np.clip(fur, 0, 1)


def ear_mask_frame(f, cat, nose_x):
    """Follows the ears with the small head-turn; used as a protect mask."""
    dx = float(nose_x - 678.0)
    c1 = (575 + 0.7 * dx, 228)
    c2 = (850 + 0.9 * dx, 232)
    e1 = sstep(1.0 - np.sqrt(((X_GRID - c1[0]) / 82) ** 2 + ((Y_GRID - c1[1]) / 68) ** 2))
    e2 = sstep(1.0 - np.sqrt(((X_GRID - c2[0]) / 86) ** 2 + ((Y_GRID - c2[1]) / 72) ** 2))
    ears = np.maximum(e1, e2)
    # Keep ear rims even if the fur classifier is weak (orange / cream tips).
    b, g, r = cv2.split(f.astype(np.float32))
    luma = 0.114 * b + 0.587 * g + 0.299 * r
    tip = np.maximum(
        sstep((r - g - 10) / 15) * sstep((r - 80) / 20),
        sstep((luma - 140) / 25) * sstep((r - b + 5) / 15),
    )
    ears = ears * sstep((np.maximum(cat, tip) - 0.08) / 0.20)
    ears = cv2.GaussianBlur(ears, (0, 0), 3.0)
    return np.clip(ears, 0, 1)


def snout_mask():
    # muzzle: nose, mouth, whisker pads — more than the eyes
    m = sstep(1.0 - np.sqrt(((X_GRID - 680) / 95) ** 2 + ((Y_GRID - 390) / 70) ** 2))
    # stay off the eyes
    m = m * (1.0 - sstep(1.0 - np.sqrt(((X_GRID - 655) / 55) ** 2 + ((Y_GRID - 315) / 40) ** 2)))
    m = m * (1.0 - sstep(1.0 - np.sqrt(((X_GRID - 755) / 55) ** 2 + ((Y_GRID - 320) / 40) ** 2)))
    return cv2.GaussianBlur(np.clip(m, 0, 1), (0, 0), 3.0)


def eyes_layer_mask():
    e1 = sstep(1.0 - np.sqrt(((X_GRID - 655) / 62) ** 2 + ((Y_GRID - 315) / 42) ** 2))
    e2 = sstep(1.0 - np.sqrt(((X_GRID - 755) / 62) ** 2 + ((Y_GRID - 322) / 42) ** 2))
    m = np.maximum(e1, e2)
    return cv2.GaussianBlur(np.clip(m, 0, 1), (0, 0), 2.5)


def mouth_restore_mask(nose_x=678.0):
    """Tight feathered mouth box. Restored from v41 (rigidly translated with the snout)."""
    dx = float(nose_x - 678.0)
    m = sstep(1.0 - np.sqrt(((X_GRID - (680.0 + dx)) / 82) ** 2 + ((Y_GRID - 396.0) / 54) ** 2))
    return cv2.GaussianBlur(np.clip(m, 0, 1), (0, 0), 2.2)


# Still-object masks on the plate (static geometry; cat is subtracted per frame)
def prop_masks():
    X, Y = X_GRID, Y_GRID
    masks = {}
    # Laptop body + bezel + keyboard only. The screen cartoon is live in v41
    # (characters move) and must not be frozen to the plate.
    laptop = sstep((X - 0) / 6) * sstep((350 - X) / 10) * sstep((Y - 360) / 8) * sstep((675 - Y) / 10)
    # Perspective screen interior (inside the bezel):
    # top y~378, bottom y~556; right edge ~x 150 at the top, ~x 198 at the bottom.
    y_s = np.clip((Y - 378.0) / (556.0 - 378.0), 0, 1)
    x_right = 150.0 + 48.0 * y_s
    screen = (sstep((X - 8) / 6) * sstep((x_right - X) / 7)
              * sstep((Y - 378) / 7) * sstep((556 - Y) / 7))
    screen = cv2.GaussianBlur(np.clip(screen, 0, 1), (0, 0), 2.0)
    laptop = np.clip(laptop - screen, 0, 1)
    masks['laptop'] = cv2.GaussianBlur(laptop, (0, 0), 3.0)
    masks['laptop_screen'] = screen  # not plated; kept for QC / docs

    # Headphones on the desk
    phones = sstep(1.0 - np.sqrt(((X - 300) / 115) ** 2 + ((Y - 615) / 70) ** 2))
    # flatten a bit into a rounded box
    phones = np.maximum(phones, sstep((X - 190) / 10) * sstep((420 - X) / 10) * sstep((Y - 555) / 8) * sstep((680 - Y) / 8) * 0.9)
    masks['headphones'] = cv2.GaussianBlur(np.clip(phones, 0, 1), (0, 0), 3.0)

    # Mic capsule + boom arm (right). Conservative: dark metallic object only later.
    mic_boom = sstep((X - 1020) / 12) * sstep((1240 - X) / 10) * sstep((Y - 130) / 12) * sstep((400 - Y) / 14)
    mic_arm = sstep((X - 980) / 10) * sstep((1200 - X) / 10) * sstep((Y - 250) / 12) * sstep((430 - Y) / 10)
    mic_cap = sstep(1.0 - np.sqrt(((X - 990) / 78) ** 2 + ((Y - 395) / 48) ** 2))
    mic = np.clip(np.maximum.reduce([mic_boom * 0.9, mic_arm * 0.85, mic_cap]), 0, 1)
    masks['mic'] = cv2.GaussianBlur(mic, (0, 0), 2.6)

    # Books: bookcase on the far right, behind the mic / ivy
    books_r = sstep((X - 1120) / 12) * sstep((1279 - X) / 3) * sstep((Y - 80) / 10) * sstep((420 - Y) / 12)
    masks['books'] = cv2.GaussianBlur(books_r, (0, 0), 3.0)

    # Plants: hanging ivy (left wall, over window, right), pots on the desk, right foreground bush.
    # Green-leaf confirmation is applied later so snow in the gaps stays live.
    ivy_l = sstep((X - 0) / 4) * sstep((230 - X) / 14) * sstep((Y - 0) / 4) * sstep((520 - Y) / 14)
    ivy_top = sstep((X - 200) / 12) * sstep((1000 - X) / 16) * sstep((Y - 0) / 3) * sstep((110 - Y) / 14)
    ivy_r = sstep((X - 1000) / 14) * sstep((1279 - X) / 4) * sstep((Y - 0) / 4) * sstep((280 - Y) / 16)
    pot_a = sstep(1.0 - np.sqrt(((X - 200) / 70) ** 2 + ((Y - 430) / 80) ** 2))
    pot_b = sstep(1.0 - np.sqrt(((X - 355) / 60) ** 2 + ((Y - 455) / 70) ** 2))
    bush_r = sstep((X - 1020) / 16) * sstep((1279 - X) / 3) * sstep((Y - 430) / 12) * sstep((719 - Y) / 6)
    plants = np.maximum.reduce([ivy_l, ivy_top, ivy_r, pot_a, pot_b, bush_r])
    masks['plants'] = cv2.GaussianBlur(np.clip(plants, 0, 1), (0, 0), 3.5)

    # Lamp fixture (still metal, listed implicitly with static lighting). Not the window.
    # Metal fixture only (shade, arm, stem) — not the wall wash under the bulb.
    shade = sstep(1.0 - np.sqrt(((X - 255) / 95) ** 2 + ((Y - 115) / 55) ** 2))
    stem = sstep((X - 70) / 8) * sstep((200 - X) / 10) * sstep((Y - 40) / 8) * sstep((360 - Y) / 10)
    lamp = np.maximum(shade, stem * 0.85) * (1.0 - window_glass_mask())
    masks['lamp'] = cv2.GaussianBlur(np.clip(lamp, 0, 1), (0, 0), 2.8)

    for k in list(masks):
        masks[k] = np.clip(masks[k], 0, 1)
    return masks


def green_leaf_weight(f):
    hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV).astype(np.float32)
    # plants in this painting are yellow-green to deep green
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    leaf = sstep((h - 28) / 8) * sstep((95 - h) / 10) * sstep((s - 40) / 25) * sstep((v - 35) / 20)
    return cv2.GaussianBlur(np.clip(leaf, 0, 1), (0, 0), 1.6)


def nose_x_of(f):
    roi = f[340:400, 600:750]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    m = ((hsv[..., 0] < 20) | (hsv[..., 0] > 160)) & (hsv[..., 1] > 40) & (hsv[..., 2] > 80)
    ys, xs = np.where(m)
    if len(xs) < 20:
        return np.nan
    return 600.0 + float(np.median(xs))


def rigid_translate(img, dx, dy):
    """Integer+subpixel rigid translate, LINEAR (no cubic ringing)."""
    M = np.array([[1, 0, dx], [0, 1, dy]], np.float32)
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def apply_layer_shift(base, layer_src, mask, dx, dy):
    """Composite a rigidly shifted copy of layer_src into base, under mask."""
    if abs(dx) < 1e-3 and abs(dy) < 1e-3:
        return base
    shifted = rigid_translate(layer_src, dx, dy)
    msh = rigid_translate(mask, dx, dy)
    a = np.clip(mask * msh, 0, 1)[..., None] if msh.ndim == 2 else np.clip(mask * msh, 0, 1)
    if a.ndim == 2:
        a = a[..., None]
    return np.clip(base * (1.0 - a) + shifted * a, 0, 255)


# --------------- blink retiming ---------------
# Real both-eye blinks in this 25s cut (cycle phase 72), plus the cycle
# wink at phase 156. 3.1s and 11.8s are NOT blinks (held squint / open);
# we still snap every real blink.
# Source close is ~n157-160, hold 160-161, open 162-166. Compress to
# 3-frame close + 1 hold + 3-frame open, nearest-frame, no crossfade.
BLINK_EVENTS = [
    # (dest start, list of source frames to paste into dest start..start+len-1)
    # dest window starts at the first closing frame.
    # 6.67s family (phase 72): original n156 is pre (leave), n157-166 is the blink
]
# We'll fill these after measuring, using a shared pattern in cycle space.
# phase 72 blink at clip i = 161 (and 161+241=402). Window n157-166.
# phase 156 wink at i=4 (and 4+241=245, 4+482=486). Window n1-8.

def blink_map():
    """dest_frame -> src_frame for the eye patch (nearest, no blend)."""
    m = {}
    # Both-eye blink, 3 close + 1 shut + 3 open
    # original: 157,158,159,160,161,162,163,164,165,166
    # new at 157-163: 157,159,160, 160, 162,164,166
    # Stay within ±2 source frames so the head pose stays compatible
    # (a +9 jump onto 166 pasted a more-frontal eye into a 3/4 socket).
    pattern_both = {
        0: 0,   # 157 -> 157
        1: 2,   # 158 -> 159  (skip smeary mid-close)
        2: 3,   # 159 -> 160  (shut)
        3: 3,   # 160 -> 160  (hold shut 1 frame)
        4: 5,   # 161 -> 162  (start open, skip extra shut)
        5: 6,   # 162 -> 163
        # 163+ stay natural so we don't paste a later, more-frontal eye
    }
    for base in (157, 157 + 241):  # 157, 398
        if base + 6 < NFR:
            for di, si in pattern_both.items():
                m[base + di] = base + si
    # Opening wink at the cut start / cycle: mostly a blink, ~n1-8, smeary open.
    # Compress to 3 close + 1 shut + 3 open, nearby donors only
    pattern_wink = {0: 0, 1: 2, 2: 3, 3: 3, 4: 5, 5: 6}
    for base in (1, 1 + 241, 1 + 482):  # 1, 242, 483
        if base + 6 < NFR:
            for di, si in pattern_wink.items():
                src = base + si
                if 0 <= src < NFR:
                    m[base + di] = src
    return m


def eye_patch_mask():
    e1 = sstep(1.0 - np.sqrt(((X_GRID - 655) / 58) ** 2 + ((Y_GRID - 318) / 40) ** 2))
    e2 = sstep(1.0 - np.sqrt(((X_GRID - 755) / 58) ** 2 + ((Y_GRID - 324) / 40) ** 2))
    m = np.maximum(e1, e2)
    # don't reach the brows / nose too far
    return cv2.GaussianBlur(np.clip(m, 0, 1), (0, 0), 2.8)


# --------------- lighting ---------------
def lighting_weight():
    """Where the static lamp layer is allowed: cat head/chest, not ears, not window."""
    head = sstep(1.0 - np.sqrt(((X_GRID - 700) / 180) ** 2 + ((Y_GRID - 370) / 160) ** 2))
    return head


def lamp_illum():
    """Screen-space lamp falloff (warm, from the fixture at top-left). Not a face shape."""
    dist = np.sqrt((X_GRID - 250.0) ** 2 + (Y_GRID - 90.0) ** 2)
    illum = np.exp(-0.5 * (dist / 300.0) ** 2)
    # slight downward cone from the shade
    illum = illum * sstep((Y_GRID - 40) / 30)
    illum = cv2.GaussianBlur(illum.astype(np.float32), (0, 0), 8.0)
    # mid-grey at the face centre so we only add a gentle left-bright / right-dark
    illum = illum - float(illum[360, 700])
    return illum


def relight(cur, illum, w):
    # Additive/multiply static layer only — do not flatten (flatten melted the muzzle).
    gain = 1.0 + 0.10 * illum
    warm = 1.0 + np.array([-0.015, 0.0, 0.02], np.float32) * (w[..., None] * (0.5 + 0.5 * np.clip(illum[..., None] / 0.3, 0, 1)))
    field = cur * gain[..., None] * warm
    a = w[..., None]
    return np.clip(cur * (1.0 - a) + field * a, 0, 255)


# --------------- main process ---------------
def process(frames, preview_indices=None):
    print('computing median plate…', flush=True)
    # subsample every 2nd frame for median (odd/even both covered)
    stack = np.stack([frames[i] for i in range(0, NFR, 2)], 0)
    plate = np.median(stack, 0).astype(np.float32)
    del stack
    cv2.imwrite(str(QC / 'plate-median.png'), plate.astype(np.uint8))

    print('building static masks…', flush=True)
    Wglass = window_glass_mask()
    Steam = steam_mask()
    Title = title_mask()
    Prior = cat_prior()
    Props = prop_masks()
    Snout = snout_mask()
    EyesL = eyes_layer_mask()
    EyePatch = eye_patch_mask()
    LightW0 = lighting_weight()
    BMap = blink_map()
    leaf_plate = green_leaf_weight(plate.astype(np.uint8))

    # nose track (smooth)
    nx = np.array([nose_x_of(f) for f in frames], np.float32)
    # fill nan
    for i in range(NFR):
        if not np.isfinite(nx[i]):
            nx[i] = nx[i - 1] if i else 678.0
    # 5-tap smooth, edge-padded so the first/last frames are not pulled toward 0
    k = np.array([1, 2, 3, 2, 1], np.float32); k /= k.sum()
    nxs = np.convolve(np.pad(nx, 2, mode='edge'), k, mode='valid')
    assert len(nxs) == NFR
    rest = float(np.median(nxs))
    turn = nxs - rest  # negative = nose left
    np.save(WORK / 'nose_x.npy', nxs)

    Illum = lamp_illum()

    # Dilated union of the cat matte across the whole span, so a plate can never
    # hold a ghost of fur in a pixel the cat occupies in ANY frame. False
    # positives on the laptop cartoon (orange/white characters outside the body
    # prior) are suppressed before the union is dilated.
    print('computing cat-union (any-frame) mask…', flush=True)
    cat_union = np.zeros((H, W), np.float32)
    for fi, fr in enumerate(frames):
        c = cat_mask_frame(fr, Prior)
        c = c * sstep((Prior - 0.12) / 0.20)
        cat_union = np.maximum(cat_union, c)
    ker_u = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    cat_any = cv2.dilate((cat_union > 0.35).astype(np.uint8), ker_u).astype(np.float32)
    cat_any = cv2.GaussianBlur(cat_any, (0, 0), 3.0)
    cv2.imwrite(str(QC / 'cat-union-dilated.png'), (np.clip(cat_any, 0, 1) * 255).astype(np.uint8))
    # drop the helper key so it is not plated
    ScreenLive = Props.pop('laptop_screen', np.zeros((H, W), np.float32))

    # save mask preview on n0 and n131
    def dump_masks(i, cat, ears, prop_a):
        vis = frames[i].copy()
        # overlays
        def mix(m, col):
            nonlocal vis
            a = np.clip(m, 0, 1)[..., None] * 0.45
            vis = (vis * (1 - a) + np.array(col) * a).astype(np.uint8)
        mix(Wglass, (0, 180, 255))
        mix(Steam, (255, 80, 0))
        mix(Title, (0, 255, 0))
        mix(cat, (0, 255, 255))
        mix(ears, (0, 0, 255))
        mix(prop_a, (255, 0, 255))
        cv2.imwrite(str(QC / f'masks-n{i}.jpg'), vis, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

    out = [None] * NFR
    targets = preview_indices if preview_indices is not None else range(NFR)
    # need source frames for blink remaps
    want = set(targets)
    want.update(BMap.values())
    want.update(BMap.keys())

    print('processing', len(list(targets)), 'frames…', flush=True)
    change_acc = np.zeros((H, W), np.float64)

    for i in range(NFR):
        if i not in want and preview_indices is not None:
            continue
        src = frames[i].astype(np.float32)
        cat = cat_mask_frame(frames[i], Prior)
        ears = ear_mask_frame(frames[i], cat, nxs[i])
        # Window glass stays live (snow) but does not claim the cat's head/ears.
        wg = Wglass * (1.0 - sstep((cat - 0.45) / 0.25))
        protect = np.clip(np.maximum.reduce([Title, wg, Steam, ears]), 0, 1)
        # window glass: also protect where it's clearly sky/snow (high-L blue) even
        # if the geometric pane missed a sliver — already in Wglass.

        # --- static plates ---
        prop_a = np.zeros((H, W), np.float32)
        for name, m in Props.items():
            a = m.copy()
            a = a * (1.0 - sstep((cat - 0.25) / 0.25))   # never cover the cat this frame
            a = a * (1.0 - sstep((cat_any - 0.12) / 0.25))  # nor any-frame fur (no ghost plate)
            a = a * (1.0 - protect)
            if name == 'plants':
                # only lock actual leaves, so snow in ivy gaps stays live
                a = a * np.maximum(leaf_plate, green_leaf_weight(frames[i]) * 0.5)
            if name == 'mic':
                # only dark metal, not the red cushion / cat cheek
                luma = 0.114 * src[..., 0] + 0.587 * src[..., 1] + 0.299 * src[..., 2]
                # cushion is dark-red (R >> B); metal is neutral-dark
                redish = sstep(((src[..., 2] - src[..., 0]) - 18) / 16)
                a = a * sstep((80 - luma) / 22) * (1.0 - 0.9 * redish)
            prop_a = np.maximum(prop_a, a)
        prop_a = cv2.GaussianBlur(np.clip(prop_a, 0, 1), (0, 0), 1.8)
        # feather already in the blur; composite plate
        comp = src * (1.0 - prop_a[..., None]) + plate * prop_a[..., None]

        # --- parallax: rigid feathered layers (no displacement-field stretch). ---
        # The previous remap of a blended dx field stretched fur under the eyes
        # and pinched the meow mouth. Each layer is now a single rigid translate,
        # dissolved in under a feathered mask. Mouth is restored from v41 later.
        dx = float(turn[i])
        extra_snout = float(np.clip(0.28 * dx, -2.0, 2.0))
        extra_eyes = float(np.clip(0.12 * dx, -1.0, 1.0))
        extra_snout_dy = 0.10 * extra_snout
        em = EyesL * cat * (1.0 - protect)
        sm = Snout * cat * (1.0 - protect) * (1.0 - EyesL * 0.85)
        # eyes first (smaller offset); snout on top so it moves more
        if abs(extra_eyes) > 0.04:
            comp = apply_layer_shift(comp, comp, em, extra_eyes, 0.0)
        if abs(extra_snout) > 0.04 or abs(extra_snout_dy) > 0.04:
            comp = apply_layer_shift(comp, comp, sm, extra_snout, extra_snout_dy)

        # --- static lamp lighting (screen-space falloff, not a median face) ---
        # Keep the eyes' own shading (pupils / lids).
        MouthM = mouth_restore_mask(nxs[i])
        lw = LightW0 * cat * (1.0 - protect) * (1.0 - EyesL * 0.85) * (1.0 - MouthM * 0.95)
        lw = cv2.GaussianBlur(lw, (0, 0), 5.0) * 0.40
        comp = relight(comp, Illum, lw)

        # --- snappier blinks: replace eye patch from a remapped source frame ---
        if i in BMap and BMap[i] != i:
            sidx = BMap[i]
            donor = frames[sidx].astype(np.float32)
            # align donor to this frame by the nose-x difference (rigid, subtle)
            ddx = float(nxs[i] - (nxs[sidx] if np.isfinite(nxs[sidx]) else nxs[i]))
            if abs(ddx) > 0.05:
                donor = rigid_translate(donor, ddx, 0.0)
            a = EyePatch * (1.0 - Title) * (1.0 - Wglass)
            # don't let the eye patch overwrite the ear
            a = a * (1.0 - ears)
            a = cv2.GaussianBlur(a, (0, 0), 1.2)
            comp = comp * (1.0 - a[..., None]) + donor * a[..., None]

        # --- mouth: pixel-identical to v41 in shape, rigidly translated with the snout ---
        if abs(extra_snout) > 0.04 or abs(extra_snout_dy) > 0.04:
            src_m = rigid_translate(src, extra_snout, extra_snout_dy)
            mm = MouthM
            mm_s = rigid_translate(mm, extra_snout, extra_snout_dy)
            ma = np.clip(np.maximum(mm, mm_s), 0, 1)
        else:
            src_m = src
            ma = MouthM
        ma = ma * (1.0 - Title) * (1.0 - Wglass)
        ma = cv2.GaussianBlur(np.clip(ma, 0, 1), (0, 0), 1.0)
        comp = comp * (1.0 - ma[..., None]) + src_m * ma[..., None]

        # --- restore protected regions from the original v41 frame ---
        # Feathered copy-back so we cannot leave a halo at the plate / layer edge
        # against the window, steam, title, or ears.
        pr = cv2.GaussianBlur(protect, (0, 0), 1.0)
        comp = comp * (1.0 - pr[..., None]) + src * pr[..., None]
        # hard-copy the title pill (must be byte-identical in the core box)
        comp[14:60, 16:165] = src[14:60, 16:165]

        out_i = np.clip(comp, 0, 255).astype(np.uint8)
        if preview_indices is None:
            d = np.any(out_i != frames[i], axis=2)
            change_acc += d
        out[i] = out_i

        if i in (0, 74, 131, 161, 245, 283, 374, 402, 426) or (preview_indices and i in preview_indices and i % 10 == 0):
            if i in (0, 131):
                dump_masks(i, cat, ears, prop_a)

        if i % 50 == 0:
            print('  frame', i, flush=True)

    if preview_indices is not None:
        return out, None, Props

    # changed-region map
    change_frac = change_acc / NFR
    return out, change_frac, Props


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


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--preview', action='store_true')
    args = ap.parse_args()

    # Refuse to proceed if someone swapped the master
    got = sha256(SRC_MP4)
    if got != V41_SHA:
        raise SystemExit(f'v41-test-25s.mp4 sha changed: {got} (expected {V41_SHA}). Aborting.')

    frames = load_src()
    if args.preview:
        idx = [0, 4, 74, 131, 157, 158, 159, 160, 161, 162, 163, 164, 165,
               206, 245, 283, 312, 374, 398, 399, 400, 401, 402, 403, 404, 426, 486]
        out, _, _ = process(frames, preview_indices=idx)
        for i in idx:
            if out[i] is None:
                continue
            cv2.imwrite(str(QC / f'preview-v41b-n{i}.png'), out[i])
            # side by side crop of face
            a = frames[i][180:500, 500:920]
            b = out[i][180:500, 500:920]
            cv2.imwrite(str(QC / f'preview-face-ba-n{i}.jpg'),
                        np.hstack([a, b]), [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        # prop crops
        for i in (0, 131, 374):
            if out[i] is None:
                continue
            crops = []
            for name, (y0, y1, x0, x1) in {
                'laptop': (380, 700, 0, 380),
                'phones': (540, 700, 180, 450),
                'mic': (280, 500, 880, 1120),
                'plantsR': (480, 720, 1040, 1280),
                'ivy': (0, 160, 400, 800),
            }.items():
                pair = np.hstack([frames[i][y0:y1, x0:x1], out[i][y0:y1, x0:x1]])
                crops.append(cv2.resize(pair, (pair.shape[1] // 2 * 2, pair.shape[0])))
                cv2.imwrite(str(QC / f'preview-prop-{name}-n{i}.jpg'), pair,
                            [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        print('preview written to', QC)
        return

    out, change_frac, Props = process(frames)
    # identity checks on protected cores
    mismatches = {}
    for i, (a, b) in enumerate(zip(frames, out)):
        if not np.array_equal(a[14:60, 16:165], b[14:60, 16:165]):
            mismatches.setdefault('title', []).append(i)
    if mismatches:
        print('WARNING protected mismatches', {k: len(v) for k, v in mismatches.items()})
    else:
        print('title core byte-identical on all frames')

    write_lossless(out, WORK / 'v41b-span-890-915-lossless.mkv')
    encode_mp4(out, P / 'v41b-test-25s.mp4')
    viz = np.clip(change_frac * 255 * 4, 0, 255).astype(np.uint8)
    cv2.imwrite(str(QC / 'changed-region-map.png'), cv2.applyColorMap(viz, cv2.COLORMAP_INFERNO))
    np.save(WORK / 'change_frac.npy', change_frac)
    print('done', sha256(P / 'v41b-test-25s.mp4'))
    # confirm master untouched
    got = sha256(SRC_MP4)
    assert got == V41_SHA, got
    print('v41 master still', got)


if __name__ == '__main__':
    main()
