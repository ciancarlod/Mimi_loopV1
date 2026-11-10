#!/usr/bin/env python3
"""v41c: defect-only pass on top of frozen v41b r2.

  - Restore crisp ear fur detail from the original v24 cycle (high-frequency
    + wispy tip hairs). Keep the v30 opaque fill (no holes). Ears still
    follow the existing head motion; no new warp.
  - Tighten ivy / desk-pot / non-right-corner leaf lock to the static plate
    (green leaf cores only) so leftover source shimmer dies, snow in the
    gaps stays live. Right-corner plant is left as in r2 (frozen, no new
    sway) until atmosphere/tail work is approved.
  - Mouth pixels in the mouth box are copied from v41b r2 (no retime here).
  - Does not apply dusk, dust, lamp flicker, tail, or plant reaction.
  - Never writes v41 or v41b masters.
"""
from __future__ import annotations
import hashlib, subprocess as sp
from pathlib import Path
import cv2
import numpy as np

P = Path('/workspace/lofi-v41-meow')
QC = P / 'qc' / 'v41c'
QC.mkdir(parents=True, exist_ok=True)
WORK = P / 'v41c'
WORK.mkdir(exist_ok=True)

SRC_R2 = P / 'v41b' / 'v41b-r2-span-890-915-lossless.mkv'
ORIG = Path('/workspace/lofi-v24/v24-original-mimi-one-cycle-mild-squint.mp4')
PCM = P / 'work' / 'test-audio-890-915-v41.f32le'
PLATE = P / 'qc' / 'v41b' / 'plate-median.png'
V41_MP4 = P / 'v41-test-25s.mp4'
V41B_R2 = P / 'v41b-r2-test-25s.mp4'
V41_SHA = 'f53cb1127c75775b740b555dd856e0002c784f80a2a3097118d8e197a794e95f'
R2_SHA = 'c1578db5782025d9236267c8c852a011e5c6e739ff82f785e62aa392d6f23b0f'
NFR, W, H, FPS, NCYCLE = 600, 1280, 720, 24, 241

def sstep(x):
    x = np.clip(x, 0, 1)
    return x * x * x * (x * (6 * x - 15) + 10)

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()

def phase_of(i):
    return (152 + i) % NCYCLE

def load_cycle(path, n=NCYCLE):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
        if len(frames) >= n:
            break
    cap.release()
    assert len(frames) == n, (path, len(frames))
    return frames

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

def title_mask():
    m = np.zeros((H, W), np.float32)
    m[14:60, 16:165] = 1.0
    return cv2.GaussianBlur(m, (0, 0), 1.2)

def window_glass_mask():
    wm = cv2.imread('/workspace/lofi-v30/safe-window-mask.png', 0).astype(np.float32) / 255.0
    pane = sstep((X - 228) / 10) * sstep((640 - X) / 10) * sstep((Y - 78) / 8) * sstep((405 - Y) / 8)
    m = np.maximum(wm, pane * 0.95)
    return np.clip(cv2.GaussianBlur(m, (0, 0), 1.6), 0, 1)

def steam_mask():
    sm = cv2.imread('/workspace/lofi-v28/steam-mask.png', 0).astype(np.float32) / 255.0
    M = np.zeros((H, W), np.float32)
    y0, x0 = 389, 394
    M[y0:y0 + sm.shape[0], x0:x0 + sm.shape[1]] = sm
    return np.clip(cv2.GaussianBlur(M, (0, 0), 2.4), 0, 1)

def mouth_mask():
    m = sstep(1.0 - np.sqrt(((X - 680.0) / 82) ** 2 + ((Y - 396.0) / 54) ** 2))
    return cv2.GaussianBlur(np.clip(m, 0, 1), (0, 0), 2.2)

def eye_mask():
    e1 = sstep(1.0 - np.sqrt(((X - 655) / 58) ** 2 + ((Y - 318) / 40) ** 2))
    e2 = sstep(1.0 - np.sqrt(((X - 755) / 58) ** 2 + ((Y - 324) / 40) ** 2))
    return cv2.GaussianBlur(np.clip(np.maximum(e1, e2), 0, 1), (0, 0), 2.8)

def ear_rois():
    # (y0,y1,x0,x1) generous boxes
    return {'L': (140, 300, 490, 670), 'R': (140, 310, 750, 940)}

def ear_color_mask(f, roi=None):
    """Orange / cream fur in an optional ROI, full-frame mask."""
    b, g, r = cv2.split(f.astype(np.float32))
    luma = 0.114 * b + 0.587 * g + 0.299 * r
    white = sstep((luma - 140) / 25) * sstep((r - b + 8) / 18)
    orange = sstep((r - 80) / 22) * sstep((r - g - 10) / 14) * sstep((r - b - 20) / 18)
    fur = np.maximum(white, orange)
    # spatial prior near the ear boxes
    prior = np.zeros((H, W), np.float32)
    for y0, y1, x0, x1 in ear_rois().values():
        prior[y0:y1, x0:x1] = 1.0
    prior = cv2.GaussianBlur(prior, (0, 0), 8.0)
    fur = fur * (0.15 + 0.85 * prior)
    fur = cv2.GaussianBlur(fur, (0, 0), 1.6)
    return np.clip(fur, 0, 1)

def shift_img(img, dx, dy):
    M = np.array([[1, 0, dx], [0, 1, dy]], np.float32)
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

def align_roi(src, dst, box):
    y0, y1, x0, x1 = box
    g0 = cv2.cvtColor(src[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float32)
    g1 = cv2.cvtColor(dst[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float32)
    # subtract means for stability
    g0 -= g0.mean(); g1 -= g1.mean()
    (dx, dy), resp = cv2.phaseCorrelate(g0, g1)
    if not np.isfinite(dx) or not np.isfinite(dy) or resp < 0.25 or abs(dx) > 8 or abs(dy) > 8:
        return 0.0, 0.0, float(resp if np.isfinite(resp) else 0)
    return float(dx), float(dy), float(resp)

def green_leaf(f):
    hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    leaf = sstep((h - 28) / 8) * sstep((95 - h) / 10) * sstep((s - 40) / 25) * sstep((v - 35) / 20)
    return cv2.GaussianBlur(np.clip(leaf, 0, 1), (0, 0), 1.4)

def foliage_lock_mask(plate, frame):
    """Green leaf cores of ivy + desk pots. Not the right-corner plant, not snow."""
    # ivy left / top / right-of-window (not the lower-right bush)
    ivy_l = sstep((X - 0) / 4) * sstep((230 - X) / 14) * sstep((Y - 0) / 4) * sstep((520 - Y) / 14)
    ivy_top = sstep((X - 200) / 12) * sstep((1000 - X) / 16) * sstep((Y - 0) / 3) * sstep((110 - Y) / 14)
    ivy_r = sstep((X - 1000) / 14) * sstep((1279 - X) / 4) * sstep((Y - 0) / 4) * sstep((280 - Y) / 16)
    pot_a = sstep(1.0 - np.sqrt(((X - 200) / 70) ** 2 + ((Y - 430) / 80) ** 2))
    pot_b = sstep(1.0 - np.sqrt(((X - 355) / 60) ** 2 + ((Y - 455) / 70) ** 2))
    geo = np.maximum.reduce([ivy_l, ivy_top, ivy_r, pot_a, pot_b])
    leaf = np.maximum(green_leaf(plate), green_leaf(frame) * 0.5)
    # erode to cores so snow/gaps stay live
    core = cv2.erode((leaf * geo > 0.35).astype(np.uint8),
                     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    a = cv2.GaussianBlur(core.astype(np.float32), (0, 0), 1.2) * geo
    return np.clip(a, 0, 1)

def restore_ears(cur, orig, Title, Wglass, Steam, Mouth, Eyes):
    """Add original-cycle high-frequency fur and wispy tips onto the filled ears."""
    cur_f = cur.astype(np.float32)
    orig_f = orig.astype(np.float32)
    out = cur_f.copy()
    fur_c = ear_color_mask(cur)
    fur_o = ear_color_mask(orig)
    protect = np.clip(np.maximum.reduce([Title, Steam, Mouth, Eyes]), 0, 1)
    # Wooden window mullion / posts: the fur classifier can catch warm wood
    # and must not rewrite the frame.
    mullion = (sstep((X - 615) / 4) * sstep((658 - X) / 4)
               * sstep((Y - 70) / 8) * sstep((420 - Y) / 8))
    mullion = np.maximum(mullion, sstep((X - 220) / 6) * sstep((250 - X) / 6)
                         * sstep((Y - 70) / 8) * sstep((420 - Y) / 8))
    # a second post often sits near x 430-460
    mullion = np.maximum(mullion, sstep((X - 425) / 5) * sstep((465 - X) / 5)
                         * sstep((Y - 70) / 8) * sstep((420 - Y) / 8))
    # post just right of the right ear
    mullion = np.maximum(mullion, sstep((X - 855) / 5) * sstep((915 - X) / 5)
                         * sstep((Y - 70) / 8) * sstep((420 - Y) / 8))
    mullion = cv2.GaussianBlur(np.clip(mullion, 0, 1), (0, 0), 1.2)
    leaf_block = green_leaf(cur)
    protect = np.clip(np.maximum.reduce([protect, mullion, leaf_block * 0.9]), 0, 1)
    for name, box in ear_rois().items():
        dx, dy, resp = align_roi(orig, cur, box)
        aligned = shift_img(orig_f, dx, dy) if (abs(dx) > 0.02 or abs(dy) > 0.02) else orig_f
        fur_a = shift_img(fur_o, dx, dy) if (abs(dx) > 0.02 or abs(dy) > 0.02) else fur_o
        y0, y1, x0, x1 = box
        # interior: high-frequency of aligned orig, only where BOTH have fur
        # (keeps the v30 fill; does not open holes)
        interior = np.clip(fur_c * fur_a, 0, 1)
        interior = cv2.GaussianBlur(interior, (0, 0), 1.0)
        interior = interior * (1.0 - protect)
        blur = cv2.GaussianBlur(aligned, (0, 0), 1.15)
        hf = aligned - blur
        # also a touch of unsharp on the current ear
        cblur = cv2.GaussianBlur(out, (0, 0), 1.0)
        unsharp = out - cblur
        a = interior[..., None]
        out = out + a * (0.70 * hf + 0.25 * unsharp)
        # wispy tips: orig fur just outside the current fur silhouette
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        dil = cv2.dilate((fur_c > 0.35).astype(np.uint8), ker).astype(np.float32)
        ero = cv2.erode((fur_c > 0.40).astype(np.uint8),
                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))).astype(np.float32)
        ring = np.clip(dil - ero, 0, 1)
        ring = cv2.GaussianBlur(ring, (0, 0), 1.0)
        wisp = ring * sstep((fur_a - 0.28) / 0.22) * (1.0 - sstep((fur_c - 0.55) / 0.2))
        # do not paint wisps over title / steam / mouth / eyes
        wisp = wisp * (1.0 - protect)
        # confine to this ear box (feathered)
        boxm = np.zeros((H, W), np.float32)
        boxm[y0:y1, x0:x1] = 1.0
        boxm = cv2.GaussianBlur(boxm, (0, 0), 4.0)
        wisp = wisp * boxm
        out = out * (1.0 - wisp[..., None]) + aligned * wisp[..., None]
    out = np.clip(out, 0, 255)
    # hard-preserve title and mouth (defect pass must not retouch the mouth box)
    out[14:60, 16:165] = cur[14:60, 16:165]
    mm = Mouth[..., None] if Mouth.ndim == 2 else Mouth
    out = out * (1.0 - mm) + cur.astype(np.float32) * mm
    return out.astype(np.uint8)

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

def process(r2, orig, preview=None):
    Title = title_mask()
    Wglass = window_glass_mask()
    Steam = steam_mask()
    Mouth = mouth_mask()
    Eyes = eye_mask()
    plate = cv2.imread(str(PLATE)).astype(np.float32)
    want = set(preview) if preview else set(range(NFR))
    out = [None] * NFR
    for i in range(NFR):
        if i not in want:
            continue
        src = r2[i]
        ph = phase_of(i)
        o = orig[ph]
        # foliage core lock (ivy + desk pots only)
        fl = foliage_lock_mask(plate.astype(np.uint8), src)
        fl = fl * (1.0 - Title) * (1.0 - Steam) * (1.0 - Mouth)
        # never lock window snow: subtract glass where it is not leaf-green
        leaf = green_leaf(src)
        fl = fl * (1.0 - Wglass * (1.0 - sstep((leaf - 0.3) / 0.25)))
        comp = src.astype(np.float32) * (1.0 - fl[..., None]) + plate * fl[..., None]
        comp = np.clip(comp, 0, 255).astype(np.uint8)
        comp = restore_ears(comp, o, Title, Wglass, Steam, Mouth, Eyes)
        # title + mouth core hard copy from r2 (mouth must stay approved)
        comp[14:60, 16:165] = src[14:60, 16:165]
        mc = (Mouth > 0.35)[..., None]
        comp = np.where(mc, src, comp)
        # Keep snow / open glass as in r2 except where current fur (or added tufts) sits
        fur_now = ear_color_mask(comp)
        glass_keep = Wglass * (1.0 - sstep((fur_now - 0.22) / 0.25)) * (1.0 - Title) * (1.0 - Steam)
        glass_keep = cv2.GaussianBlur(np.clip(glass_keep, 0, 1), (0, 0), 0.8)
        comp = (comp.astype(np.float32) * (1.0 - glass_keep[..., None])
                + src.astype(np.float32) * glass_keep[..., None])
        comp = np.clip(comp, 0, 255).astype(np.uint8)
        comp[14:60, 16:165] = src[14:60, 16:165]
        out[i] = comp
        if i % 50 == 0:
            print('  frame', i, flush=True)
    return out

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--preview', action='store_true')
    args = ap.parse_args()

    got_v41 = sha256(V41_MP4)
    if got_v41 != V41_SHA:
        raise SystemExit(f'v41 master sha changed: {got_v41}')
    got_r2 = sha256(V41B_R2)
    if got_r2 != R2_SHA:
        raise SystemExit(f'v41b-r2 sha changed: {got_r2}')

    print('loading r2 span + orig cycle…', flush=True)
    r2 = load_span(SRC_R2)
    orig = load_cycle(ORIG)
    if args.preview:
        idx = [0, 4, 70, 114, 131, 133, 145, 151, 191, 226, 283, 374, 384, 426, 430, 440]
        out = process(r2, orig, preview=idx)
        for i in idx:
            cv2.imwrite(str(QC / f'preview-n{i}.png'), out[i])
            # ear BA
            a = r2[i][140:300, 490:940]
            b = out[i][140:300, 490:940]
            o = orig[phase_of(i)][140:300, 490:940]
            trip = np.hstack([a, b, o])
            cv2.imwrite(str(QC / f'preview-ears-n{i}.jpg'), trip, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            # mouth should match r2
            md = np.abs(out[i][350:440, 620:760].astype(np.int16) - r2[i][350:440, 620:760].astype(np.int16)).max()
            print(f'  n{i} mouth maxabs vs r2 = {md}')
        print('preview written', QC)
        return

    out = process(r2, orig)
    # title identity
    bad = [i for i, (a, b) in enumerate(zip(r2, out)) if not np.array_equal(a[14:60, 16:165], b[14:60, 16:165])]
    print('title mismatches' if bad else 'title core byte-identical', len(bad) if bad else '')
    mouth_bad = 0
    for a, b in zip(r2, out):
        # core mouth ellipse should stay very close (we copy by mask; bilinear edge ok)
        d = np.abs(a[360:430, 640:740].astype(np.int16) - b[360:430, 640:740].astype(np.int16)).max()
        if d > 0:
            mouth_bad += 1
    print('mouth-core changed frames (expect 0):', mouth_bad)
    write_lossless(out, WORK / 'v41c-span-890-915-lossless.mkv')
    encode_mp4(out, P / 'v41c-test-25s.mp4')
    print('done', sha256(P / 'v41c-test-25s.mp4'))
    print('v41 still', sha256(V41_MP4))
    print('r2 still', sha256(V41B_R2))

if __name__ == '__main__':
    main()
