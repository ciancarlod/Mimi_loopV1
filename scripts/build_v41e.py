#!/usr/bin/env python3
"""v41e: atmosphere lighting pass on frozen v41d.

Recipes 1, 2, 3, 4, 8 from /workspace/lofi-research/LOFI-RESEARCH.md:
  late-dusk window, darker room with lamp as key, lamp flicker WITHOUT
  stacking a new 3-cycle pulse on the existing v28 pulse, one-cycle
  ambiance breath, subtle grain + lamp bloom.

Dust, tail idle, and the right-corner plant brush are deferred to v41f
so the grade can be checked first (mouths, snow, title, luma maps).

Does not overwrite v41 / v41b / v41c / v41d. PCM is the approved v41 stem.
"""
from __future__ import annotations
import hashlib, subprocess as sp, sys
from pathlib import Path
import cv2
import numpy as np

P = Path('/workspace/lofi-v41-meow')
QC = P / 'qc' / 'v41e'
QC.mkdir(parents=True, exist_ok=True)
(QC / 'preview').mkdir(exist_ok=True)
(QC / 'encoded').mkdir(exist_ok=True)
WORK = P / 'v41e'
WORK.mkdir(exist_ok=True)

SRC = P / 'v41d' / 'v41d-span-890-915-lossless.mkv'
PCM = P / 'work' / 'test-audio-890-915-v41.f32le'
V41_MP4 = P / 'v41-test-25s.mp4'
V41C_MP4 = P / 'v41c-test-25s.mp4'
V41D_MP4 = P / 'v41d-test-25s.mp4'
V41_SHA = 'f53cb1127c75775b740b555dd856e0002c784f80a2a3097118d8e197a794e95f'
V41C_SHA = 'a752e0545cf8886fde322490c25da5bb45f9d715e0dfda312959bfb99bb29fb9'
V41D_SHA = 'd1dab25187f4756969344036f50cd0ef89fac7c414d80006b8853acf04be2b34'
PCM_SHA = 'a635074f164c14fb3faabbf33c258703e9b7b9645eddbf8028f63a29573745a0'
NFR, W, H, FPS, NCYCLE = 600, 1280, 720, 24, 241

sys.path.insert(0, str(P))
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
# Lamp fixture (from v41b / v28): shade centred near (255, 115).
LX, LY = 255.0, 115.0

def hex_bgr(h):
    h = h.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return np.array([b, g, r], np.float32)

def luma(im):
    return 0.114 * im[..., 0] + 0.587 * im[..., 1] + 0.299 * im[..., 2]

def dusk_sky_color(y, w_breath):
    """Vertical late-dusk ramp, plus a one-cycle afterglow breath.
    y is 0 at the top of the frame.
    w_breath is 0 at ph 0/241 and 1 at mid-loop.
    """
    # Window pane roughly y 78–405. Afterglow sits LOW (bottom 10–18% of pane).
    t = np.clip((y - 78.0) / (405.0 - 78.0), 0, 1)
    upper = hex_bgr('0D1440')
    mid = hex_bgr('171A46')
    blend = hex_bgr('3A2434')
    # horizon gold-rose, crushed; mid-loop shifts slightly toward amber
    rose = hex_bgr('C47A6A')
    amber = hex_bgr('D4A565')
    horizon = rose * (1.0 - 0.45 * w_breath) + amber * (0.45 * w_breath)
    # lum targets (afterglow ± ~12 points across the breath)
    # Use colour as written; scale later.
    # Bands: upper 0–0.45, mid 0.45–0.82, afterglow 0.82–0.92, horizon lip 0.92–1
    c = np.zeros(y.shape + (3,), np.float32)
    # vectorized lerp through stops
    stops_t = np.array([0.0, 0.40, 0.78, 0.90, 1.0], np.float32)
    # horizon lip brighter at mid-breath
    h_gain = 0.72 + 0.28 * w_breath
    cols = np.stack([
        upper,
        mid,
        blend,
        horizon * (0.55 + 0.15 * w_breath),  # short afterglow band
        horizon * h_gain * 0.85,
    ], 0)
    # np.interp per channel
    for k in range(3):
        c[..., k] = np.interp(t, stops_t, cols[:, k])
    return c

def lamp_falloff():
    """Warm pool from the table lamp; steep near, weak far. World-locked."""
    dx = X - LX
    dy = Y - LY
    # downward cone
    r = np.sqrt((dx / 210.0) ** 2 + ((dy - 40) / 280.0) ** 2)
    core = np.exp(-0.5 * (r / 0.45) ** 2)
    mid = np.exp(-0.5 * (r / 0.95) ** 2)
    pool = 0.70 * core + 0.30 * mid
    # little upward leak above the shade
    pool = pool * (0.35 + 0.65 * sstep((Y - 70) / 40.0))
    return cv2.GaussianBlur(np.clip(pool, 0, 1), (0, 0), 8.0)

def shade_core_mask():
    m = sstep(1.0 - np.sqrt(((X - LX) / 70) ** 2 + ((Y - LY) / 40) ** 2))
    return cv2.GaussianBlur(np.clip(m, 0, 1), (0, 0), 3.0)

def make_grain(n=NCYCLE, seed=41):
    rng = np.random.RandomState(seed)
    # slightly larger than a frame so we can unique-shift per phase if needed
    vol = rng.randn(n, H, W).astype(np.float32)
    # mild blur so it isn't video snow; mono + tiny warm tint applied later
    for i in range(n):
        vol[i] = cv2.GaussianBlur(vol[i], (0, 0), 0.55)
    # normalise to sigma ~ 1
    vol /= (vol.std() + 1e-6)
    return vol

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


def process(src, grain, preview=None):
    Title = C.title_mask()
    Wglass = C.window_glass_mask()
    Steam = C.steam_mask()
    Mouth = C.mouth_mask()
    Eyes = C.eye_mask()
    Face = np.clip(np.maximum(Mouth, Eyes), 0, 1)
    Face = cv2.GaussianBlur(np.clip(Face + C.sstep(1.0 - np.sqrt(((X - 700) / 130) ** 2 + ((Y - 360) / 160) ** 2)), 0, 1), (0, 0), 6.0)
    Pool = lamp_falloff()
    Shade = shade_core_mask()
    # Mullions stay wooden — grade the glass, not the posts.
    Mull = np.zeros((H, W), np.float32)
    for x0, x1 in ((220, 250), (425, 465), (615, 658), (855, 915)):
        Mull = np.maximum(Mull, sstep((X - x0) / 4) * sstep((x1 - X) / 4)
                          * sstep((Y - 70) / 8) * sstep((420 - Y) / 8))
    Mull = cv2.GaussianBlur(np.clip(Mull, 0, 1), (0, 0), 1.2)
    # Do not punch guessed mullion strips out of the grade (that left day-sky
    # bands). The safe window mask already follows the posts. Exclude fur so
    # ear tufts in the pane are not re-graded as sky.
    glass = np.clip(Wglass * (1.0 - Title) * (1.0 - Steam * 0.4), 0, 1)

    # Cool room fill (far / window side darker than the lamp pool).
    far = 1.0 - sstep((Pool - 0.18) / 0.25)
    far = cv2.GaussianBlur(far, (0, 0), 10.0)

    want = set(preview) if preview else set(range(NFR))
    out = [None] * NFR
    for i in range(NFR):
        if i not in want:
            continue
        ph = phase_of(i)
        # one-cycle breath, 0 at ph 0 and 241, 1 at mid-loop
        w = 0.5 - 0.5 * np.cos(2.0 * np.pi * ph / NCYCLE)
        # 7-cycle flicker only (do not add a new 3-cycle term on the v28 pulse)
        flick = 1.0 + 0.005 * np.sin(2.0 * np.pi * 7 * ph / NCYCLE)

        img = src[i].astype(np.float32)
        L0 = luma(img)
        fur = C.ear_color_mask(src[i])
        # Only real ear/cheek fur, not the ear-ROI prior (that prior is a rectangle
        # and was punching a gray hole in the pane beside the left ear).
        glass_i = np.clip(glass * (1.0 - sstep((fur - 0.55) / 0.20)), 0, 1)

        # --- 1. Late-dusk window: grade the low-frequency sky, keep snow residual ---
        sky = dusk_sky_color(Y, w)
        # Low-pass of the current pane (sky + trees); residual is snow / fine twigs.
        pane = img.copy()
        low = cv2.GaussianBlur(pane, (0, 0), 7.0)
        residual = pane - low
        # Mix the dusk ramp into the low frequencies, preserve a bit of tree structure.
        # Trees go darker/cooler by leaning the low pass toward the ramp.
        low_l = luma(low)[..., None]
        # Avoid dividing by tiny luma
        dusk_low = sky * 0.55 + low * (40.0 / np.maximum(low_l, 20.0)) * 0.35
        # Afterglow breath also slightly warms the residual snow
        snow_gain = 0.90 + 0.08 * w
        dusk_pane = dusk_low + residual * snow_gain
        g = glass_i[..., None]
        img = img * (1.0 - g) + dusk_pane * g

        # Weak cool glass fill into the room near the pane (always dimmer than the lamp).
        fill = hex_bgr('222844')
        fill_amt = (0.10 + 0.02 * (w - 0.5)) * (1.0 - Pool) * (1.0 - Title)
        fill_amt = fill_amt * sstep((X - 180) / 80.0) * sstep((700 - X) / 120.0) * sstep((Y - 60) / 40.0) * sstep((520 - Y) / 80.0)
        fill_amt = cv2.GaussianBlur(np.clip(fill_amt, 0, 1), (0, 0), 16.0)
        img = img * (1.0 - 0.35 * fill_amt)[..., None] + fill * (0.35 * fill_amt)[..., None]

        # --- 2. Darker room, lamp stays the key ---
        # Cool multiply on the far room; much weaker in the lamp pool and on the face.
        # Target: far floor lum well below the shade, global pulled down, window not the key.
        cool = np.array([1.02, 0.92, 0.82], np.float32)  # BGR, cooler in the shadows
        dark_amt = 0.38 * far * (1.0 - 0.65 * Face) * (1.0 - Title)
        # Don't crush the dusk window we just graded (already dark).
        dark_amt = dark_amt * (1.0 - 0.85 * glass[..., 0] if glass.ndim == 3 else 1.0 - 0.85 * glass)
        dark_amt = cv2.GaussianBlur(np.clip(dark_amt, 0, 1), (0, 0), 8.0)
        darkened = img * cool * (0.52 + 0.48 * Pool)[..., None]
        # Face keeps more of its current (lamp-lit) value so the mouth stays readable.
        mix = dark_amt * (1.0 - 0.55 * Face)
        img = img * (1.0 - mix)[..., None] + darkened * mix[..., None]

        # Warm the lamp pool (shade + bounce on nearby desk). World-locked.
        warm = np.array([0.92, 1.00, 1.10], np.float32)
        boost = (0.16 * Pool + 0.22 * Shade) * flick
        img = img * (1.0 + boost * 0.55)[..., None] * (1.0 + (warm - 1.0) * (0.35 * Pool)[..., None])

        # --- 8/2. Lamp bloom (core only), flickered, not on the face ---
        core = (Shade * 0.75 + Pool * 0.25) * (1.0 - 0.85 * Face) * (1.0 - Title)
        core_b = cv2.GaussianBlur(core, (0, 0), 28.0)
        bloom_c = hex_bgr('FFE6B0')
        bloom_op = (0.18 * flick) * core_b
        img = img + bloom_c * bloom_op[..., None]  # add; clip later
        # A tighter core glow
        core2 = cv2.GaussianBlur(Shade * (1.0 - Title), (0, 0), 12.0)
        img = img + bloom_c * ((0.10 * flick) * core2)[..., None]

        # Room bounce from the afterglow (30–50% of the window change): very weak
        # warm lift on far walls at mid-loop, must not flatten the lamp contrast.
        bounce = 0.04 * w * far * (1.0 - Pool) * (1.0 - Face) * (1.0 - Title)
        bounce = cv2.GaussianBlur(np.clip(bounce, 0, 1), (0, 0), 20.0)
        img = img + hex_bgr('5A2E32') * (0.12 * bounce)[..., None]

        # --- 8. Looping grain, lighter on the face and mouth ---
        nmap = grain[ph]
        # optional 2-frame analog hold: even/odd share... 241 is odd, so use ph//1 (unique per phase) 
        # slightly warm mono grain
        gcol = nmap[..., None] * np.array([0.96, 1.00, 1.04], np.float32)
        gop = 0.10 * (1.0 - 0.75 * Face) * (1.0 - 0.5 * Title)
        # soft-light-ish: img + grain * sigma, sigma ~ 2.8 gray
        img = img + gcol * (2.8 * gop)[..., None]

        # Title stays the approved scrolling card (readability).
        img = np.clip(img, 0, 255)
        img[14:60, 16:165] = src[i][14:60, 16:165]
        # Mouth: do not reshape; a light grade is ok, but keep it from washing out.
        # Already protected via Face in the darken/bloom. No hard copy (would undo dusk on whisker edges).

        out[i] = img.astype(np.uint8)
        if i % 25 == 0 or i in (133, 138, 430, 432, 0, 120):
            Yi = luma(out[i])
            print(f'  n{i} ph={ph} w={w:.2f} glob={Yi.mean():.1f} shade={Yi[80:200,180:330].mean():.1f} '
                  f'win={Yi[90:360,240:600].mean():.1f} far={Yi[560:700,900:1200].mean():.1f} '
                  f'mouth={Yi[360:430,640:740].mean():.1f}', flush=True)
    return out


def save_preview(src, out, idx):
    q = QC / 'preview'
    for i in idx:
        if out[i] is None:
            continue
        cv2.imwrite(str(q / f'full-n{i}.jpg'), out[i], [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        a, b = src[i][350:470, 600:780], out[i][350:470, 600:780]
        cv2.imwrite(str(q / f'mouth-n{i}.jpg'), np.hstack([a, b]), [int(cv2.IMWRITE_JPEG_QUALITY), 93])
        # luma map
        Y = luma(out[i].astype(np.float32))
        heat = cv2.applyColorMap(np.clip(Y, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        cv2.imwrite(str(q / f'luma-n{i}.jpg'), np.hstack([out[i], heat]), [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        # window crop
        a, b = src[i][70:420, 210:660], out[i][70:420, 210:660]
        cv2.imwrite(str(q / f'window-n{i}.jpg'), np.hstack([a, b]), [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    print('preview written', q)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--preview', action='store_true')
    args = ap.parse_args()

    for label, path, expect in (
        ('v41', V41_MP4, V41_SHA),
        ('v41c', V41C_MP4, V41C_SHA),
        ('v41d', V41D_MP4, V41D_SHA),
        ('pcm', PCM, PCM_SHA),
    ):
        got = sha256(path)
        if got != expect:
            raise SystemExit(f'{label} sha changed: {got}')
    print('frozen shas ok', flush=True)
    print('loading v41d lossless…', flush=True)
    src = load_span(SRC)
    print('building grain volume…', flush=True)
    grain = make_grain()

    if args.preview:
        idx = [0, 60, 120, 180, 240,
               133, 138, 145,
               191, 370, 374,
               430, 432, 440]
        out = process(src, grain, preview=idx)
        save_preview(src, out, idx)
        return

    out = process(src, grain)
    bad = [i for i, (a, b) in enumerate(zip(src, out))
           if not np.array_equal(a[14:60, 16:165], b[14:60, 16:165])]
    print('title mismatches' if bad else 'title core byte-identical', len(bad) if bad else '')
    los = WORK / 'v41e-r2-span-890-915-lossless.mkv'
    mp4 = P / 'v41e-r2-test-25s.mp4'
    print('writing lossless…', flush=True)
    write_lossless(out, los)
    print('encoding mp4…', flush=True)
    encode_mp4(out, mp4)
    print('v41e-r2 mp4', sha256(mp4))
    print('v41 still', sha256(V41_MP4))
    print('v41d still', sha256(V41D_MP4))

if __name__ == '__main__':
    main()
