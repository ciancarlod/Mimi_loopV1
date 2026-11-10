#!/usr/bin/env python3
"""Per-frame cat silhouettes for v41f via rembg u2net.
Keeps the largest component in the cat region (drops the lamp blob).
Temporal median smooth + dilate/erode. Writes a grayscale ffv1 span.
"""
import sys, time, subprocess as sp
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
from rembg import remove, new_session

SRC = Path('/workspace/lofi-v41-meow/v41d/v41d-span-890-915-lossless.mkv')
OUT = Path('/workspace/lofi-v41-meow/v41f/cat-alpha-890-915.mkv')
QC = Path('/workspace/lofi-v41-meow/qc/v41f/mattes')
QC.mkdir(parents=True, exist_ok=True)
NFR, W, H = 600, 1280, 720

def cat_component(alpha):
    """Keep the cat blob; drop the lamp shade blob in the upper left."""
    a = alpha.astype(np.float32) / 255.0
    # Zero the known lamp region so it cannot join the cat
    a[0:250, 0:480] = 0
    bin = (a > 0.35).astype(np.uint8)
    n, lab, stats, cents = cv2.connectedComponentsWithStats(bin, 8)
    if n <= 1:
        return a
    # pick the component with largest area whose centroid is in the cat half
    best, best_area = 0, 0
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        cx, cy = cents[i]
        if area < 2000:
            continue
        if cx < 400 or cy < 150:
            continue
        if area > best_area:
            best, best_area = i, area
    if best == 0:
        # fallback: largest overall after lamp punch
        best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    keep = (lab == best).astype(np.float32)
    # recover original alpha on that component, plus a little halo already in rembg
    out = a * keep
    # close small holes
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    hard = cv2.morphologyEx((out > 0.4).astype(np.uint8), cv2.MORPH_CLOSE, ker)
    out = np.maximum(out, cv2.GaussianBlur(hard.astype(np.float32), (0, 0), 1.6) * 0.6)
    return np.clip(out, 0, 1)

def main():
    only = None
    if '--only' in sys.argv:
        only = set(int(x) for x in sys.argv[sys.argv.index('--only')+1].split(','))
    print('loading session…', flush=True)
    session = new_session('u2net')
    cap = cv2.VideoCapture(str(SRC))
    raw = []
    t0 = time.time()
    for i in range(NFR):
        ok, f = cap.read()
        assert ok
        if only is not None and i not in only:
            raw.append(None)
            continue
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(rgb)
        rgba = remove(im, session=session)
        alpha = np.array(rgba)[..., 3]
        raw.append(cat_component(alpha))
        if i % 50 == 0:
            print(f'  rembg {i}/{NFR}  {time.time()-t0:.1f}s', flush=True)
    cap.release()
    if only is not None:
        for i in sorted(only):
            a = (np.clip(raw[i], 0, 1) * 255).astype(np.uint8)
            cv2.imwrite(str(QC / f'cat-n{i}.png'), a)
        print('preview mattes written', QC)
        return
    # temporal median of 5, with edge pad
    print('temporal smooth…', flush=True)
    arr = np.stack(raw, 0)  # N,H,W float
    sm = np.empty_like(arr)
    for i in range(NFR):
        sl = arr[max(0, i-2):min(NFR, i+3)]
        sm[i] = np.median(sl, 0)
    # slight dilate for a protect mask? store the smooth silhouette (feathered)
    # A 2px dilate is applied at use time; here just close and feather
    ker3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    out_u8 = []
    for i in range(NFR):
        a = sm[i]
        hard = cv2.morphologyEx((a > 0.4).astype(np.uint8), cv2.MORPH_CLOSE, ker3)
        a = np.maximum(a, hard.astype(np.float32) * 0.5)
        a = cv2.GaussianBlur(np.clip(a, 0, 1), (0, 0), 1.2)
        out_u8.append((a * 255).astype(np.uint8))
        if i in (0, 60, 120, 138, 180, 370, 432):
            cv2.imwrite(str(QC / f'cat-n{i}.png'), out_u8[i])
    enc = sp.Popen([
        'ffmpeg', '-v', 'error', '-y',
        '-f', 'rawvideo', '-pix_fmt', 'gray', '-s', f'{W}x{H}', '-r', '24', '-i', '-',
        '-an', '-c:v', 'ffv1', '-level', '3', str(OUT)
    ], stdin=sp.PIPE)
    for a in out_u8:
        enc.stdin.write(a.tobytes())
    enc.stdin.close()
    assert enc.wait() == 0
    print('wrote', OUT, 'elapsed', time.time()-t0)

if __name__ == '__main__':
    main()
