import cv2, numpy as np, json, subprocess as sp
from pathlib import Path
P = Path('/workspace/lofi-v41-meow')
prm = json.load(open(P / 'cycle' / 'v41-medium-params.json'))
boxes = {r['frame']: r['box'] for r in prm['records']}
eboxes = {r['frame']: r['expanded_box'] for r in prm['records']}
v40 = cv2.VideoCapture('/workspace/lofi-v40-meow/v40-test-25s.mp4')
v41 = cv2.VideoCapture(str(P / 'v41-test-25s.mp4'))
NFR = 600

def style_label(i, t, ver):
    if 123 <= i <= 151: tag = 'OPEN mid'
    elif 179 <= i <= 207: tag = 'MRRRP'
    elif 364 <= i <= 392: tag = 'PURR'
    elif 420 <= i <= 448: tag = 'OPEN big'
    else: tag = ''
    return f'{ver} {tag}  {t:.2f}s'

def eye_panel(frame, ebox, label, w=640, h=720):
    # 3x on the eyes: top portion of the head ebox, centered on the eye line.
    xa, ya, ew, eh = ebox
    # eyes sit in the top ~100 px of the ebox
    cw, ch = w // 3, h // 3
    cx = xa + ew // 2
    cy = ya + 48  # eye row
    xa2 = int(np.clip(cx - cw / 2, 0, frame.shape[1] - cw))
    ya2 = int(np.clip(cy - ch / 2, 0, frame.shape[0] - ch))
    crop = frame[ya2:ya2 + ch, xa2:xa2 + cw]
    if crop.shape[0] != ch or crop.shape[1] != cw:
        canvas = np.zeros((ch, cw, 3), np.uint8)
        canvas[:crop.shape[0], :crop.shape[1]] = crop
        crop = canvas
    z = cv2.resize(crop, (w, h), interpolation=cv2.INTER_CUBIC)
    cv2.putText(z, label, (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(z, label, (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (240, 240, 240), 1, cv2.LINE_AA)
    return z

enc = sp.Popen([
    'ffmpeg', '-v', 'error', '-y',
    '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', '1280x720', '-r', '24', '-i', '-',
    '-c:v', 'libx264', '-preset', 'slow', '-crf', '14', '-pix_fmt', 'yuv420p',
    '-an', '-movflags', '+faststart',
    str(P / 'v40-vs-v41-eyes-before-after.mp4')
], stdin=sp.PIPE)
nfr = 0
for i in range(NFR):
    ok40, f40 = v40.read(); ok41, f41 = v41.read()
    assert ok40 and ok41, i
    ph = (890 * 24 + i) % 241
    ebox = eboxes.get(ph, [540, 266, 250, 272])
    t = i / 24
    pair = np.concatenate([
        eye_panel(f40, ebox, style_label(i, t, 'v40')),
        eye_panel(f41, ebox, style_label(i, t, 'v41'))
    ], 1)
    enc.stdin.write(pair.tobytes())
    nfr += 1
enc.stdin.close(); assert enc.wait() == 0
print('wrote eyes before-after', nfr, 'frames')
