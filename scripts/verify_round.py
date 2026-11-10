#!/usr/bin/env python3
"""Encoded-frame verification for a round: crops, seams, audio, stats."""
import sys, subprocess as sp, json
from pathlib import Path
import cv2, numpy as np
P = Path('/workspace/lofi-v41-meow')
ver = sys.argv[1]
mp4 = P / f'{ver}-test-25s.mp4'
dest = P / 'qc' / ver / 'encoded'
dest.mkdir(parents=True, exist_ok=True)
W, H = 1280, 720
def frames(path):
    p = sp.run(['ffmpeg', '-v', 'error', '-i', str(path), '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-'], capture_output=True, check=True)
    return np.frombuffer(p.stdout, np.uint8).reshape(-1, H, W, 3)
F = frames(mp4)
print('frames', len(F))
L = lambda im: im.astype(np.float32) @ np.array([0.114, 0.587, 0.299], np.float32)
cr = {'tail': (480, 720, 920, 1280, 3), 'sky': (0, 205, 250, 860, 2), 'window': (0, 420, 250, 860, 2),
      'face': (200, 520, 520, 900, 2), 'mouth': (360, 480, 640, 780, 4), 'lamp': (40, 280, 200, 500, 2),
      'plant': (430, 720, 1000, 1280, 2), 'earL': (150, 270, 540, 680, 4), 'earR': (150, 280, 820, 960, 4),
      'desk': (300, 620, 100, 700, 2), 'laptop': (370, 600, 0, 210, 2)}
idx = [int(x) for x in (sys.argv[2].split(',') if len(sys.argv) > 2 else '0,60,88,89,120,133,138,150,180,199,241,370,374,430,432,440,599'.split(','))]
for i in idx:
    f = F[i]
    cv2.imwrite(str(dest / f'full-n{i}.png'), f)
    for tag, (y0, y1, x0, x1, s) in cr.items():
        cv2.imwrite(str(dest / f'{tag}-n{i}.png'), cv2.resize(f[y0:y1, x0:x1], None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST))
st = {}
st['glob_mean'] = float(np.mean([L(F[i]).mean() for i in range(0, 600, 10)]))
st['face_mean'] = float(np.mean([L(F[i])[300:340, 700:740].mean() for i in range(0, 600, 10)]))
st['lamp_mean'] = float(np.mean([L(F[i])[120:170, 280:360].mean() for i in range(0, 600, 10)]))
st['far_room'] = float(np.mean([L(F[i])[60:300, 1100:1270].mean() for i in range(0, 600, 10)]))
st['sky_top'] = float(np.mean([L(F[i])[10:60, 400:540].mean() for i in range(0, 600, 10)]))
# seams: same-phase n0 vs n241, n1 vs n242 and wrap n88->n89 vs typical neighbour diff
d = lambda a, b: float(np.abs(F[a].astype(np.int16) - F[b].astype(np.int16)).mean())
st['n0_vs_n241'] = d(0, 241); st['n1_vs_n242'] = d(1, 242); st['n100_vs_n341'] = d(100, 341)
st['n88_n89'] = d(88, 89); st['n329_n330'] = d(329, 330)
nb = [d(i, i + 1) for i in range(0, 599, 3)]
st['nbr_median'] = float(np.median(nb)); st['nbr_p95'] = float(np.percentile(nb, 95)); st['nbr_max'] = float(np.max(nb))
gl = [L(F[i]).mean() for i in range(600)]
st['glob_ptp'] = float(np.ptp(gl)); st['glob_maxstep'] = float(np.max(np.abs(np.diff(gl))))
lp = [L(F[i])[100:260, 200:480].mean() for i in range(600)]
st['lamp_ptp'] = float(np.ptp(lp)); st['lamp_maxstep'] = float(np.max(np.abs(np.diff(lp))))
# laptop screen liveliness and right plant motion and other foliage
def mad(a, b, box):
    y0, y1, x0, x1 = box
    return float(np.abs(F[a][y0:y1, x0:x1].astype(np.int16) - F[b][y0:y1, x0:x1].astype(np.int16)).mean())
st['laptop_n0_n60'] = mad(0, 60, (395, 560, 10, 190))
st['desk_pots_n0_n60'] = mad(0, 60, (330, 440, 130, 330))
st['ivy_left_n0_n60'] = mad(0, 60, (0, 300, 0, 200))
st['right_plant_n189_n199'] = mad(189, 199, (520, 720, 1080, 1280))
# window snow liveliness
st['window_n0_n6'] = mad(0, 6, (100, 380, 300, 540))
# audio
def pcm(path):
    p = sp.run(['ffmpeg', '-v', 'error', '-i', str(path), '-vn', '-f', 'f32le', '-ac', '2', '-ar', '48000', '-'], capture_output=True, check=True)
    return np.frombuffer(p.stdout, np.float32)
a = pcm(P / 'v41-test-25s.mp4'); b = pcm(mp4)
st['aac_maxabs_vs_v41'] = float(np.max(np.abs(a - b))) if len(a) == len(b) else f'len {len(a)} vs {len(b)}'
pr = sp.run(['ffprobe', '-v', 'error', '-show_entries', 'stream=codec_name,width,height,r_frame_rate,nb_frames,bit_rate,pix_fmt', '-of', 'json', str(mp4)], capture_output=True, text=True)
st['probe'] = json.loads(pr.stdout)
print(json.dumps(st, indent=1))
(dest / 'stats.json').write_text(json.dumps(st, indent=1))
